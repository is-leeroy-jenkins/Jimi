'''
  ******************************************************************************************
      Assembly:                Jimi
      Filename:                boogr.py
      Author:                  Terry D. Eppler
      Created:                 05-31-2022

      Last Modified By:        Terry D. Eppler
      Last Modified On:        05-01-2025
  ******************************************************************************************
  <copyright file="boogr.py" company="Terry D. Eppler">

	     boogr.py
	     Copyright ©  2025  Terry Eppler

     Permission is hereby granted, free of charge, to any person obtaining a copy
     of this software and associated documentation files (the “Software”),
     to deal in the Software without restriction,
     including without limitation the rights to use,
     copy, modify, merge, publish, distribute, sublicense,
     and/or sell copies of the Software,
     and to permit persons to whom the Software is furnished to do so,
     subject to the following conditions:

     The above copyright notice and this permission notice shall be included in all
     copies or substantial portions of the Software.

     THE SOFTWARE IS PROVIDED “AS IS”, WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED,
     INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
     FITNESS FOR A PARTICULAR PURPOSE AND NON-INFRINGEMENT.
     IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM,
     DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE,
     ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER
     DEALINGS IN THE SOFTWARE.

     You can contact me at:  terryeppler@gmail.com or eppler.terry@epa.gov

  </copyright>
  <summary>
    boogr.py
  </summary>
  ******************************************************************************************
'''
from __future__ import annotations

import os
import sqlite3
import traceback
from datetime import datetime
from pathlib import Path
from sys import exc_info
from typing import Any, List

import config as cfg

HEADLESS = (
		"STREAMLIT_SERVER_RUNNING" in os.environ
		or "streamlit" in os.environ.get( "PYTHONPATH", "" ).lower( )
)

class Error( Exception ):
	"""Wrap application exceptions with stable diagnostic metadata.

	Purpose:
		Captures the original exception, traceback text, optional display heading, logical
		cause, module name, and stable method signature used by Jimi exception handlers. The
		wrapper provides a consistent exception object that can be raised by calling code and
		persisted by Logger without requiring each handler to format diagnostic text manually.

	Attributes:
		exception (Exception): Original exception instance being wrapped.
		heading (str | None): Optional display heading associated with the exception.
		cause (str | None): Logical component, class, or workflow where the exception occurred.
		method (str | None): Stable method or function signature where the exception occurred.
		module (str | None): Source module where the exception occurred.
		type (type | None): Active exception type reported by sys.exc_info.
		trace (str): Formatted traceback text captured during wrapper construction.
		info (str): Combined exception type and formatted traceback text.
	"""
	
	def __init__( self, error: Exception, heading: str = None, cause: str = None,
			method: str = None, module: str = None ):
		"""Initialize the error wrapper.

		Purpose:
			Stores the original exception and optional diagnostic metadata, then captures the
			active traceback immediately. Capturing traceback state during construction preserves
			the original failure context for downstream logging and re-raising.

		Args:
			error (Exception): Original exception instance being wrapped.
			heading (str): Optional display heading associated with the exception.
			cause (str): Logical component, class, or workflow where the exception occurred.
			method (str): Stable method or function signature where the exception occurred.
			module (str): Source module where the exception occurred.
		"""
		super( ).__init__( )
		self.exception = error
		self.heading = heading
		self.cause = cause
		self.method = method
		self.module = module
		self.type = exc_info( )[ 0 ]
		self.trace = traceback.format_exc( )
		self.info = str( exc_info( )[ 0 ] ) + ': \r\n \r\n' + traceback.format_exc( )
	
	def __str__( self ) -> str | None:
		"""Return captured diagnostic text.

		Purpose:
			Returns the formatted exception information captured when the wrapper was created.
			The representation supports direct display, debugging, and logging without requiring
			callers to inspect each diagnostic field separately.

		Returns:
			Captured exception information when available.
		"""
		if self.info is not None:
			return self.info
	
	def __dir__( self ) -> List[ str ] | None:
		"""Return public diagnostic member names.

		Purpose:
			Provides a stable member list for debuggers, inspectors, documentation tooling, and
			interactive sessions that need to discover the diagnostic fields exposed by the
			exception wrapper.

		Returns:
			Public diagnostic member names exposed by the wrapper.
		"""
		return [ 'message', 'cause', 'method', 'module', 'scaler', 'stack_trace', 'info' ]

class Logger( ):
	"""Persist wrapped exception records to SQLite.

	Purpose:
		Writes Error metadata to the SQLite logging database identified by config.LOG_PATH and
		the exception table identified by config.LOG_FILE. The logger creates the logging
		directory and table as needed, then records cause, module, method, message, diagnostic
		information, traceback text, and creation time for later troubleshooting.

	Attributes:
		path (Path): Filesystem path to the SQLite logging database.
		table (str): SQLite table name used for exception records.
		query (str | None): SQL statement prepared for the active setup or write operation.
		values (tuple[Any, ...] | None): SQL parameter values prepared for the active write.
	"""
	
	def __init__( self ) -> None:
		"""Initialize the logger.

		Purpose:
			Reads the logging database path and exception table name from central configuration
			and prepares local state for later setup and write operations. The constructor does
			not open a persistent SQLite connection; each database operation owns its connection
			scope.
		"""
		self.path = Path( cfg.LOG_PATH ).resolve( )
		self.table = str( cfg.LOG_FILE or 'Exceptions' )
		self.query = None
		self.values = None
	
	def __dir__( self ) -> List[ str ]:
		"""Return public logger member names.

		Purpose:
			Provides a stable member list for inspection and debugging of logger configuration,
			including the database path, table name, prepared SQL state, and write helpers.

		Returns:
			Public logger member names.
		"""
		return [ 'path', 'table', 'query', 'values', 'create_table', 'write' ]
	
	def create_table( self ) -> None:
		"""Create the configured exception table.

		Purpose:
			Ensures the logging directory and SQLite exception table exist before an exception
			record is written. Setup failures are suppressed so logging infrastructure cannot
			mask the original application exception.
		"""
		try:
			self.path.parent.mkdir( parents=True, exist_ok=True )
			self.query = f'''
				CREATE TABLE IF NOT EXISTS {self.table} (
					id INTEGER PRIMARY KEY AUTOINCREMENT,
					created TEXT,
					cause TEXT,
					module TEXT,
					method TEXT,
					message TEXT,
					info TEXT,
					trace TEXT
				)
			'''
			with sqlite3.connect( self.path ) as connection:
				connection.execute( self.query )
				connection.commit( )
		except Exception:
			return None
	
	def write( self, error: Error ) -> None:
		"""Write an error record.

		Purpose:
			Persists a wrapped Error object to the configured SQLite database using Jimi's
			standard exception schema. Write failures are suppressed so a database or filesystem
			problem during logging does not replace the original application exception.

		Args:
			error (Error): Wrapped exception object containing diagnostic metadata to persist.
		"""
		try:
			self.create_table( )
			message = str( getattr( error, 'exception', '' ) )
			self.query = f'''
				INSERT INTO {self.table} (
					created,
					cause,
					module,
					method,
					message,
					info,
					trace
				)
				VALUES (?, ?, ?, ?, ?, ?, ?)
			'''
			self.values = (
					datetime.now( ).isoformat( timespec='seconds' ),
					getattr( error, 'cause', None ),
					getattr( error, 'module', None ),
					getattr( error, 'method', None ),
					message,
					getattr( error, 'info', None ),
					getattr( error, 'trace', None ),
			)
			with sqlite3.connect( self.path ) as connection:
				connection.execute( self.query, self.values )
				connection.commit( )
		except Exception:
			return None