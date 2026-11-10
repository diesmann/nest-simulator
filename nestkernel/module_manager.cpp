/*
 *  module_manager.cpp
 *
 *  This file is part of NEST.
 *
 *  Copyright (C) 2004 The NEST Initiative
 *
 *  NEST is free software: you can redistribute it and/or modify
 *  it under the terms of the GNU General Public License as published by
 *  the Free Software Foundation, either version 2 of the License, or
 *  (at your option) any later version.
 *
 *  NEST is distributed in the hope that it will be useful,
 *  but WITHOUT ANY WARRANTY; without even the implied warranty of
 *  MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 *  GNU General Public License for more details.
 *
 *  You should have received a copy of the GNU General Public License
 *  along with NEST.  If not, see <http://www.gnu.org/licenses/>.
 *
 */

#include "module_manager.h"

#ifdef HAVE_LIBLTDL

// Includes from libnestutil:
#include "logging.h"

// Includes from nestkernel:
#include "kernel_manager.h"
#include "nest_extension_interface.h"

// Includes from thirdparty:
#include "compose.hpp"

// Includes from C/C++:
#include <dlfcn.h>
#include <fstream>


namespace nest
{

ModuleManager::ModuleManager()
  : modules_()
{
  lt_dlinit();

  // Add NEST lib install dir to ltdl search path
  // To avoid problems due to string substitution in NEST binaries during
  // Conda installation, we need to convert the literal to string, cstr and back,
  // see #2237 and https://github.com/conda/conda-build/issues/1674#issuecomment-280378336
  const std::string module_dir = std::string( NEST_INSTALL_PREFIX ).c_str() + std::string( "/" NEST_INSTALL_LIBDIR );
  if ( lt_dladdsearchdir( module_dir.c_str() ) )
  {
    LOG( VerbosityLevel::ERROR,
      "ModuleManager::ModuleManager",
      String::compose( "Could not add dynamic module search directory '%1'.", module_dir ) );
  }
}

ModuleManager::~ModuleManager()
{
  finalize( /* adjust_number_of_threads_or_rng_only */ false );  // closes dynamically loaded modules
  lt_dlexit();
}

void
ModuleManager::initialize( const bool )
{
}

void
ModuleManager::finalize( const bool adjust_number_of_threads_or_rng_only )
{
  if ( adjust_number_of_threads_or_rng_only )
  {
    return;
  }

  // unload all loaded modules and remember those the system keeps in memory
  for ( const auto& [ name, module_info ] : modules_ )
  {
    lt_dlclose( module_info.handle );
#ifdef RTLD_NOLOAD
    if ( void* resident = dlopen( module_info.path.c_str(), RTLD_LAZY | RTLD_NOLOAD ) )
    {
      dlclose( resident );
      resident_[ module_info.path ] = module_info.file_content;
    }
    else
    {
      resident_.erase( module_info.path );
    }
#else
    // without RTLD_NOLOAD the system cannot tell whether the library left memory; assume it did not
    resident_[ module_info.path ] = module_info.file_content;
#endif
  }
  modules_.clear();
}

ModuleManager::FileContent_
ModuleManager::file_content_( const std::string& path )
{
  std::ifstream file( path, std::ios::binary );
  std::uint64_t hash = 14695981039346656037ULL;
  std::size_t size = 0;
  for ( char c; file.get( c ); ++size )
  {
    hash = ( hash ^ static_cast< unsigned char >( c ) ) * 1099511628211ULL;
  }
  return { size, hash };
}

void
ModuleManager::reinitialize_dynamic_modules()
{
  for ( const auto& [ name, module_info ] : modules_ )
  {
    module_info.extension->initialize();
  }
}

void
ModuleManager::get_status( Dictionary& d )
{
  std::vector< std::string > loaded;
  for ( const auto& [ name, module_info ] : modules_ )
  {
    loaded.emplace_back( name );
  }
  d[ names::modules ] = loaded;
}

void
ModuleManager::set_status( const Dictionary& )
{
}

void
ModuleManager::install( const std::string& name )
{
  // We cannot have connections without network elements, so we only need to check nodes.
  // Simulating an empty network causes no problems, so we don't have to check for that.
  if ( kernel().node_manager.size() > 0 )
  {
    throw KernelException(
      "Network elements have been created, so external modules can no longer be imported. "
      "Call ResetKernel() first." );
  }

  if ( name.empty() )
  {
    throw DynamicModuleManagementError( "Module name must not be empty." );
  }

  if ( modules_.find( name ) != modules_.end() )
  {
    throw DynamicModuleManagementError( "Module '" + name + "' is loaded already." );
  }

  // call lt_dlerror() to reset any error messages hanging around
  lt_dlerror();

  // try to open the module
  const lt_dlhandle hModule = lt_dlopenext( name.c_str() );

  if ( not hModule )
  {
    char* errstr = ( char* ) lt_dlerror();
    std::string msg = "Module '" + name + "' could not be opened.";
    if ( errstr )
    {
      msg += "\nThe dynamic loader returned the following error: '" + std::string( errstr ) + "'.";
    }
    msg += "\n\nPlease check LD_LIBRARY_PATH (OSX: DYLD_LIBRARY_PATH)!";
    throw DynamicModuleManagementError( msg );
  }

  // refuse a changed file whose earlier version the system kept in memory: the loader returned the old code
  const std::string path = lt_dlgetinfo( hModule )->filename;
  const auto file_content = file_content_( path );
  if ( const auto res = resident_.find( path ); res != resident_.end() and res->second != file_content )
  {
    lt_dlclose( hModule );
    lt_dlerror();
    throw DynamicModuleManagementError( "Module file '" + path
      + "' changed after an earlier version was loaded in this process, and the system kept that version in "
        "memory. Installing it again would use the earlier version. Restart NEST, or install the changed module "
        "under a new name." );
  }

  // see if we can find the "module" symbol in the module
  NESTExtensionInterface* extension = reinterpret_cast< NESTExtensionInterface* >( lt_dlsym( hModule, "module" ) );
  char* errcstr = ( char* ) lt_dlerror();
  if ( errcstr )
  {
    std::string errstr( errcstr );
    lt_dlclose( hModule );  // close module again
    lt_dlerror();           // remove any error caused by lt_dlclose()
    throw DynamicModuleManagementError(
            "Module '" + name + "' could not be loaded: could not find "
            "module symbol in module.\n"
            "The dynamic loader returned the following error: '"
            + errstr + "'.");
  }

  // all is well and we can register module components
  try
  {
    extension->initialize();
  }
  catch ( std::exception& e )
  {
    lt_dlclose( hModule );
    lt_dlerror();  // remove any error caused by lt_dlclose()
    throw;         // no arg re-throws entire exception, see Stroustrup 14.3.1
  }

  // add the handle to list of loaded modules
  modules_[ name ] = ModuleMapEntry_( hModule, extension, path, file_content );

  LOG( VerbosityLevel::INFO, "Install", ( "loaded module " + name ).c_str() );
}

}  // namespace nest

#endif /* HAVE_LIBLTDL */
