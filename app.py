'''
  ******************************************************************************************
      Assembly:                Jimi
      Filename:                app.py
      Author:                  Terry D. Eppler
      Created:                 05-31-2022

      Last Modified By:        Terry D. Eppler
      Last Modified On:        12-27-2025
  ******************************************************************************************
  <copyright file="app.py" company="Terry D. Eppler">

	     app.py
	     Copyright ©  2022  Terry Eppler

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
    app.py
  </summary>
  ******************************************************************************************
'''
from __future__ import annotations
import socket
from google import genai
from gemini import Chat
import base64
import hashlib
import re
import sqlite3
import time
from pathlib import Path
from typing import Any, Dict, List, Tuple
import numpy as np
import pandas as pd
import plotly.express as px
import streamlit as st
import config as cfg

# ==============================================================================
# Deferred Dependency Resolution
# ==============================================================================
MODEL_PATH_OBJ = Path( cfg.MODEL_PATH )

def local_model_available( ) -> bool:
	"""
		Purpose:
		--------
		Determine whether the configured local GGUF model exists.

		Parameters:
		-----------
		None

		Returns:
		--------
		bool
			True when the configured model file exists; otherwise False.
	"""
	try:
		return MODEL_PATH_OBJ.exists( )
	except Exception:
		return False

# ==============================================================================
# SESSION STATE INITIALIZATION
# ==============================================================================
if 'mode' not in st.session_state:
	st.session_state[ 'mode' ] = ''

if 'messages' not in st.session_state:
	st.session_state[ 'messages' ] = [ ]

if 'system_instructions' not in st.session_state:
	st.session_state[ 'system_instructions' ] = ''

if 'context_window' not in st.session_state:
	st.session_state[ 'context_window' ] = 0

if 'cpu_threads' not in st.session_state:
	st.session_state[ 'cpu_threads' ] = 0

if 'max_tokens' not in st.session_state:
	st.session_state[ 'max_tokens' ] = 0

if 'temperature' not in st.session_state:
	st.session_state[ 'temperature' ] = 0.0

if 'top_percent' not in st.session_state:
	st.session_state[ 'top_percent' ] = 0.0

if 'top_k' not in st.session_state:
	st.session_state[ 'top_k' ] = 0

if 'frequency_penalty' not in st.session_state:
	st.session_state[ 'frequency_penalty' ] = 0.0

if 'presence_penalty' not in st.session_state:
		st.session_state[ 'presence_penalty' ] = 0.0

if 'generation_active' not in st.session_state:
	st.session_state[ 'generation_active' ] = False

if 'generation_stop_requested' not in st.session_state:
	st.session_state[ 'generation_stop_requested' ] = False

if 'generation_request_id' not in st.session_state:
	st.session_state[ 'generation_request_id' ] = 0

if 'generation_provider' not in st.session_state:
	st.session_state[ 'generation_provider' ] = 'local'

if 'generation_status' not in st.session_state:
	st.session_state[ 'generation_status' ] = 'idle'

if 'gemini_grounding_model' not in st.session_state:
	st.session_state[ 'gemini_grounding_model' ] = 'gemini-2.5-flash-lite'

if 'gemini_grounding_available' not in st.session_state:
	st.session_state[ 'gemini_grounding_available' ] = False

if 'gemini_grounding_error' not in st.session_state:
	st.session_state[ 'gemini_grounding_error' ] = ''

if 'repeat_penalty' not in st.session_state:
	st.session_state[ 'repeat_penalty' ] = 0.0

if 'repeat_window' not in st.session_state:
	st.session_state[ 'repeat_window' ] = 0

if 'random_seed' not in st.session_state:
	st.session_state[ 'random_seed' ] = 0

if 'basic_docs' not in st.session_state:
	st.session_state[ 'basic_docs' ] = [ ]

if 'use_semantic' not in st.session_state:
	st.session_state[ 'use_semantic' ] = False

if 'is_grounded' not in st.session_state:
	st.session_state[ 'is_grounded' ] = False

if 'selected_prompt_id' not in st.session_state:
	st.session_state[ 'selected_prompt_id' ] = ''

if 'pending_system_prompt_name' not in st.session_state:
	st.session_state[ 'pending_system_prompt_name' ] = ''
	
# -------- TEXT GENERATION EXTENSIONS ---------------------

if 'task_preset' not in st.session_state:
	st.session_state[ 'task_preset' ] = 'Chat'

if 'response_format' not in st.session_state:
	st.session_state[ 'response_format' ] = 'Markdown'

if 'use_chat_history' not in st.session_state:
	st.session_state[ 'use_chat_history' ] = True

if 'use_document_context' not in st.session_state:
	st.session_state[ 'use_document_context' ] = False

if 'reasoning_depth' not in st.session_state:
	st.session_state[ 'reasoning_depth' ] = 'Medium'

if 'answer_only' not in st.session_state:
	st.session_state[ 'answer_only' ] = False

if 'use_self_check' not in st.session_state:
	st.session_state[ 'use_self_check' ] = False

if 'deterministic_reasoning' not in st.session_state:
	st.session_state[ 'deterministic_reasoning' ] = False

if 'coding_language' not in st.session_state:
	st.session_state[ 'coding_language' ] = 'Python'

if 'coding_task' not in st.session_state:
	st.session_state[ 'coding_task' ] = 'Generate'

if 'coding_include_comments' not in st.session_state:
	st.session_state[ 'coding_include_comments' ] = True

if 'coding_editor_format' not in st.session_state:
	st.session_state[ 'coding_editor_format' ] = True

if 'coding_fenced_output' not in st.session_state:
	st.session_state[ 'coding_fenced_output' ] = True

if 'translation_target_language' not in st.session_state:
	st.session_state[ 'translation_target_language' ] = 'English'

if 'active_prompt_caption' not in st.session_state:
	st.session_state[ 'active_prompt_caption' ] = ''

if 'preview_effective_prompt' not in st.session_state:
	st.session_state[ 'preview_effective_prompt' ] = False
	
# -------- DOCQNA ---------------------

if 'uploaded' not in st.session_state:
	st.session_state[ 'uploaded' ] = [ ]
	
if 'doc_file_uploader_revision' not in st.session_state:
	st.session_state[ 'doc_file_uploader_revision' ] = 0
	
if 'active_docs' not in st.session_state:
	st.session_state[ 'active_docs' ] = [ ]

if 'doc_bytes' not in st.session_state:
	st.session_state[ 'doc_bytes' ] = { }

if 'doc_source' not in st.session_state:
	st.session_state[ 'doc_source' ] = 'uploadlocal'

if 'docqna_vec_ready' not in st.session_state:
	st.session_state[ 'docqna_vec_ready' ] = False

if 'docqna_fingerprint' not in st.session_state:
	st.session_state[ 'docqna_fingerprint' ] = ''

if 'docqna_chunk_count' not in st.session_state:
	st.session_state[ 'docqna_chunk_count' ] = 0

if 'docqna_fallback_rows' not in st.session_state:
	st.session_state[ 'docqna_fallback_rows' ] = [ ]
	
if 'retrieval_k' not in st.session_state:
	st.session_state[ 'retrieval_k' ] = 6

if 'retrieval_chunk_size' not in st.session_state:
	st.session_state[ 'retrieval_chunk_size' ] = 1200

if 'retrieval_chunk_overlap' not in st.session_state:
	st.session_state[ 'retrieval_chunk_overlap' ] = 200

if 'show_retrieved_chunks' not in st.session_state:
	st.session_state[ 'show_retrieved_chunks' ] = True

if 'require_grounding' not in st.session_state:
	st.session_state[ 'require_grounding' ] = True

if 'answer_from_excerpts_only' not in st.session_state:
	st.session_state[ 'answer_from_excerpts_only' ] = True

if 'prefer_sqlite_vec' not in st.session_state:
	st.session_state[ 'prefer_sqlite_vec' ] = True

if 'allow_similarity_fallback' not in st.session_state:
	st.session_state[ 'allow_similarity_fallback' ] = True

if 'docqna_task' not in st.session_state:
	st.session_state[ 'docqna_task' ] = 'Ask a Question'

if 'ocr_enabled' not in st.session_state:
	st.session_state[ 'ocr_enabled' ] = False

if 'prefer_native_pdf_text' not in st.session_state:
	st.session_state[ 'prefer_native_pdf_text' ] = True

if 'include_page_markers' not in st.session_state:
	st.session_state[ 'include_page_markers' ] = False

if 'show_doc_parse_diagnostics' not in st.session_state:
	st.session_state[ 'show_doc_parse_diagnostics' ] = False

if 'doc_last_retrieval_hits' not in st.session_state:
	st.session_state[ 'doc_last_retrieval_hits' ] = [ ]

if 'doc_inventory_rows' not in st.session_state:
	st.session_state[ 'doc_inventory_rows' ] = [ ]

if 'doc_compare_mode' not in st.session_state:
	st.session_state[ 'doc_compare_mode' ] = False

# -------- SEMANTIC SEARCH  ---------------------

if 'semantic_chunk_size' not in st.session_state:
	st.session_state[ 'semantic_chunk_size' ] = 1200

if 'semantic_chunk_overlap' not in st.session_state:
	st.session_state[ 'semantic_chunk_overlap' ] = 200

if 'semantic_top_k' not in st.session_state:
	st.session_state[ 'semantic_top_k' ] = 8

if 'semantic_min_similarity' not in st.session_state:
	st.session_state[ 'semantic_min_similarity' ] = 0.0

if 'semantic_group_by_document' not in st.session_state:
	st.session_state[ 'semantic_group_by_document' ] = False

if 'semantic_clear_existing' not in st.session_state:
	st.session_state[ 'semantic_clear_existing' ] = True

if 'semantic_append_existing' not in st.session_state:
	st.session_state[ 'semantic_append_existing' ] = False

if 'semantic_show_diagnostics' not in st.session_state:
	st.session_state[ 'semantic_show_diagnostics' ] = True

if 'semantic_uploaded_names' not in st.session_state:
	st.session_state[ 'semantic_uploaded_names' ] = [ ]

if 'semantic_result_rows' not in st.session_state:
	st.session_state[ 'semantic_result_rows' ] = [ ]

if 'semantic_selected_rows' not in st.session_state:
	st.session_state[ 'semantic_selected_rows' ] = [ ]

if 'semantic_index_chunk_count' not in st.session_state:
	st.session_state[ 'semantic_index_chunk_count' ] = 0

if 'semantic_index_dim' not in st.session_state:
	st.session_state[ 'semantic_index_dim' ] = 0

if 'semantic_index_doc_count' not in st.session_state:
	st.session_state[ 'semantic_index_doc_count' ] = 0

if 'semantic_last_query' not in st.session_state:
	st.session_state[ 'semantic_last_query' ] = ''

if 'semantic_context_buffer' not in st.session_state:
	st.session_state[ 'semantic_context_buffer' ] = [ ]

# ----- Prompt Engineering -----

if 'pe_page' not in st.session_state:
	st.session_state[ 'pe_page' ] = 1

if 'pe_search' not in st.session_state:
	st.session_state[ 'pe_search' ] = ''

if 'pe_filter_category' not in st.session_state:
	st.session_state[ 'pe_filter_category' ] = ''

if 'pe_sort_col' not in st.session_state:
	st.session_state[ 'pe_sort_col' ] = 'ID'

if 'pe_sort_dir' not in st.session_state:
	st.session_state[ 'pe_sort_dir' ] = 'ASC'

if 'pe_selected_id' not in st.session_state:
	st.session_state[ 'pe_selected_id' ] = None

if 'pe_caption' not in st.session_state:
	st.session_state[ 'pe_caption' ] = ''

if 'pe_name' not in st.session_state:
	st.session_state[ 'pe_name' ] = ''

if 'pe_edit_category' not in st.session_state:
	st.session_state[ 'pe_edit_category' ] = ''

if 'pe_text' not in st.session_state:
	st.session_state[ 'pe_text' ] = ''

if 'pe_task_type' not in st.session_state:
	st.session_state[ 'pe_task_type' ] = 'Chat'

if 'pe_response_format' not in st.session_state:
	st.session_state[ 'pe_response_format' ] = 'Markdown'

if 'pe_language' not in st.session_state:
	st.session_state[ 'pe_language' ] = 'English'

if 'pe_generator_category' not in st.session_state:
	st.session_state[ 'pe_generator_category' ] = ''

if 'pe_generator_goal' not in st.session_state:
	st.session_state[ 'pe_generator_goal' ] = ''

if 'pe_generator_constraints' not in st.session_state:
	st.session_state[ 'pe_generator_constraints' ] = ''

if 'pe_generator_style' not in st.session_state:
	st.session_state[ 'pe_generator_style' ] = 'Practical'

if 'pe_generated_template' not in st.session_state:
	st.session_state[ 'pe_generated_template' ] = ''

if 'pe_cascade_enabled' not in st.session_state:
	st.session_state[ 'pe_cascade_enabled' ] = False

if 'pe_jump_id' not in st.session_state:
	st.session_state[ 'pe_jump_id' ] = 1

if 'pe_last_search' not in st.session_state:
	st.session_state[ 'pe_last_search' ] = ''

if 'pe_last_filter_category' not in st.session_state:
	st.session_state[ 'pe_last_filter_category' ] = ''
	
# ----- Prompt Mode Constants -----

TEXT_GENERATION_PROMPT_CATEGORIES: List[ str ] = [ 'Research / Academic', 'Prompt Engineering',
	'Writing / Administrative', 'Compliance / Legal / Budget', 'Business / Finance / Marketing',
	'Software Engineering', 'Data Analytics & Governance', 'Instruction/ Training / Planning',
	'Translation API', ]

DOCUMENT_QNA_PROMPT_CATEGORIES: List[ str ] = [ 'Research / Academic', 'Writing / Administrative',
	'Compliance / Legal / Budget', 'Business / Finance / Marketing', 'Software Engineering',
	'Data Analytics & Governance', 'Instruction/ Training / Planning', 'Translation API', ]

PROMPT_PAGE_SIZE: int = 10

ALL_CATEGORIES_LABEL: str = 'All Categories'

PROMPT_SORT_COLUMNS: List[ str ] = [ 'ID', 'Caption', 'Name', 'Category', 'Text', ]

PROMPT_TASK_TYPES: List[ str ] = [ 'Chat', 'Reasoning', 'Coding', 'Translation',
	'Summarization', 'Extraction', ]

PROMPT_RESPONSE_FORMATS: List[ str ] = [ 'Plain Text', 'Markdown', 'Bullet Summary', 'JSON', ]

PROMPT_GENERATOR_STYLES: List[ str ] = [ 'Practical', 'Formal', 'Analytical', 'Concise', ]

TASK_PRESET_DEFAULTS: Dict[ str, Any ] = { 'task_preset': 'Chat', 'response_format': 'Markdown',
	'use_chat_history': True, 'use_document_context': False,
	'translation_target_language': 'English', }

REASONING_CONTROL_DEFAULTS: Dict[ str, Any ] = { 'reasoning_depth': 'Medium', 'answer_only': False,
	'use_self_check': False, 'deterministic_reasoning': False, }

CODING_CONTROL_DEFAULTS: Dict[ str, Any ] = { 'coding_language': 'Python',
	'coding_task': 'Generate', 'coding_include_comments': True, 'coding_editor_format': True,
	'coding_fenced_output': True, }

TASK_PRESET_OPTIONS: List[ str ] = [ 'Chat', 'Reasoning', 'Coding', 'Translation', 'Summarization',
	'Extraction', ]

RESPONSE_FORMAT_OPTIONS: List[ str ] = [ 'Plain Text', 'Markdown', 'Bullet Summary', 'JSON', ]

REASONING_DEPTH_OPTIONS: List[ str ] = [ 'Low', 'Medium', 'High', ]

CODING_LANGUAGE_OPTIONS: List[ str ] = [ 'Python', 'C', 'C++', 'C#', 'Java', 'JavaScript',
	'TypeScript', 'SQL', 'VBA', 'HTML5', 'CSS3', 'Markdown', ]

CODING_TASK_OPTIONS: List[ str ] = [ 'Generate', 'Refactor', 'Explain', 'Debug', 'Review', ]

RETRIEVAL_CONTROL_DEFAULTS: Dict[ str, Any ] = { 'retrieval_k': 6, 'retrieval_chunk_size': 1200,
	'retrieval_chunk_overlap': 200, 'show_retrieved_chunks': True, 'require_grounding': True,
	'answer_from_excerpts_only': True, 'prefer_sqlite_vec': True,
	'allow_similarity_fallback': True, }

DOCUMENT_TASK_OPTIONS: List[ str ] = [
			'Ask a Question',
			'Summarize',
			'Key Points',
			'Outline',
			'Entities',
			'Tables',
			'Compare',
		]

DOCUMENT_TASK_PLACEHOLDERS: Dict[ str, str ] = {
	'Ask a Question': 'Ask a question about the active document source…',
	'Summarize': 'Optional: specify the scope, topics, or level of detail…',
	'Key Points': 'Optional: specify the findings, requirements, or decisions to emphasize…',
	'Outline': 'Optional: specify the desired outline depth or organizational focus…',
	'Entities': 'Optional: specify entity types such as people, organizations, dates, or laws…',
	'Tables': 'Optional: specify which tables or tabular facts to extract…',
	'Compare': 'Optional: specify the documents, topics, or criteria to compare…', }

DOCUMENT_TASK_CAPTIONS: Dict[ str, str ] = {
	'Ask a Question': 'Enter a question about the active document source.',
	'Summarize': 'Generate a grounded summary. Enter optional instructions or run the default '
	             'task.',
	'Key Points': 'Extract principal findings, requirements, decisions, and supporting facts.',
	'Outline': 'Generate a structured outline reflecting the document organization.',
	'Entities': 'Extract named entities and identify their roles or relevance.',
	'Tables': 'Identify and extract tables or material presented in tabular form.',
	'Compare': 'Compare at least two active documents using optional comparison criteria.', }

# ==============================================================================
# UTILITIES
# ==============================================================================

def image_to_base64( path: str ) -> str:
	with open( path, "rb" ) as f:
		return base64.b64encode( f.read( ) ).decode( )

def cosine_similarity( a: np.ndarray, b: np.ndarray ) -> float:
	denom = np.linalg.norm( a ) * np.linalg.norm( b )
	return float( np.dot( a, b ) / denom ) if denom else 0.0

def initialize_database( ) -> None:
	"""Database initialization.

	Purpose:
	    Creates the application SQLite database and required tables when they do not
	    exist. Validates that the Prompts table contains the schema required by the
	    System Instructions and Prompt Engineering workflows.

	Args:
	    None.

	Returns:
	    None: This function performs its work through side effects and does not return
	    a value.

	Raises:
	    RuntimeError: Raised when the Prompts table does not match the required schema.
	"""
	Path( 'stores/sqlite' ).mkdir( parents=True, exist_ok=True )
	required_prompt_columns: List[ str ] = [ 'ID', 'Caption', 'Name', 'Category', 'Text', ]
	with sqlite3.connect( cfg.DB_PATH ) as conn:
		conn.execute( """
                      CREATE TABLE IF NOT EXISTS "chat_history"
                      (
                          "id"
                          INTEGER
                          PRIMARY
                          KEY
                          AUTOINCREMENT,
                          "role"
                          TEXT,
                          "content"
                          TEXT
                      );
		              """ )
		
		conn.execute( """
                      CREATE TABLE IF NOT EXISTS "embeddings"
                      (
                          "id"
                          INTEGER
                          PRIMARY
                          KEY
                          AUTOINCREMENT,
                          "chunk"
                          TEXT,
                          "vector"
                          BLOB
                      );
		              """ )
		
		conn.execute( """
                      CREATE TABLE IF NOT EXISTS "Prompts"
                      (
                          "ID"
                          INTEGER
                          NOT
                          NULL
                          UNIQUE,
                          "Caption"
                          TEXT
                      (
                          80
                      ),
                          "Name" TEXT
                      (
                          80
                      ),
                          "Category" TEXT
                      (
                          80
                      ),
                          "Text" TEXT
                      (
                          2048
                      ),
                          PRIMARY KEY
                      (
                          "ID"
                          AUTOINCREMENT
                      )
                          );
		              """ )
		
		conn.commit( )
		
		prompt_schema_rows: List[ Tuple[ Any, ... ] ] = conn.execute(
			'PRAGMA table_info("Prompts");' ).fetchall( )
		
		prompt_columns: List[ str ] = [ str( row[ 1 ] ) for row in prompt_schema_rows if
			len( row ) > 1 ]
		
		missing_columns: List[ str ] = [ column for column in required_prompt_columns if
			column not in prompt_columns ]
		
		unexpected_columns: List[ str ] = [ column for column in prompt_columns if
			column not in required_prompt_columns ]
		
		if missing_columns or unexpected_columns:
			error_parts: List[ str ] = [ 'The Prompts table does not match the required schema.',
				f'Expected columns: {", ".join( required_prompt_columns )}.',
				f'Actual columns: {", ".join( prompt_columns ) or "None"}.', ]
			
			if missing_columns:
				error_parts.append( f'Missing columns: {", ".join( missing_columns )}.' )
			
			if unexpected_columns:
				error_parts.append( f'Unexpected columns: {", ".join( unexpected_columns )}.' )
			
			raise RuntimeError( ' '.join( error_parts ) )
		
		# ==================================================================================
		# PRIMARY KEY VALIDATION
		# ==================================================================================
		id_schema_row: Tuple[ Any, ... ] | None = next(
			(row for row in prompt_schema_rows if len( row ) > 1 and str( row[ 1 ] ) == 'ID'),
			None, )
		
		if id_schema_row is None:
			raise RuntimeError( 'The Prompts table does not define the required ID column.' )
		
		id_column_type: str = str( id_schema_row[ 2 ] or '' ).upper( )
		id_primary_key_position: int = int( id_schema_row[ 5 ] or 0 )
		
		if id_column_type != 'INTEGER':
			raise RuntimeError( 'The Prompts.ID column must use the INTEGER data type.' )
		
		if id_primary_key_position != 1:
			raise RuntimeError( 'The Prompts.ID column must be the table primary key.' )

# ----- Gemini Utilities

def gemini_grounding_available( ) -> bool:
	"""Determine whether Gemini grounding is configured.

	Purpose:
		Determines whether the Gemini API credential required by the existing
		Gemini wrapper is available to the application.

	Returns:
		bool: True when a nonempty Gemini API key is configured; otherwise False.
	"""
	return bool( cfg.GEMINI_API_KEY and str( cfg.GEMINI_API_KEY ).strip( ) )

def run_grounded_gemini_turn( user_input: str, model: str, stream: bool,
	output: Any | None = None, ) -> str:
	"""Execute one Google-grounded Gemini Text Generation turn.

	Purpose:
		Builds the effective Text Generation prompt, resolves the shared generation
		parameters, and submits the request through the existing Gemini Chat wrapper
		with Google Search enabled. The function supports streamed and non-streamed
		responses without recreating credential, client, tool, or response-processing
		logic already implemented by the wrapper.

	Args:
		user_input: Current Text Generation user request.
		model: Gemini model identifier used for the grounded request.
		stream: Indicates whether response text should be streamed.
		output: Optional Streamlit output container used during streaming.

	Returns:
		str: Grounded Gemini response text.

	Raises:
		ValueError: Raised when the prompt, API key, or model is unavailable.
		Exception: Re-raised when the Gemini wrapper reports a provider failure.
	"""
	user_input_value: str = str( user_input or '' ).strip( )
	
	if not user_input_value:
		return ''
	
	if not gemini_grounding_available( ):
		raise ValueError( 'Gemini grounding requires GEMINI_API_KEY in config.py.' )
	
	model_value: str = str( model or st.session_state.get( 'gemini_grounding_model',
		'gemini-2.5-flash-lite', ) ).strip( )
	
	if not model_value:
		raise ValueError( 'A Gemini grounding model is required.' )
	
	generation_parameters: Dict[ str, Any ] = resolve_generation_parameters( )
	
	effective_prompt: str = build_prompt( user_input=user_input_value, )
	
	chat: Chat = Chat( model=model_value, )
	
	if model_value not in (chat.model_options or [ ]):
		raise ValueError( f'Unsupported Gemini grounding model: {model_value}' )
	
	st.session_state[ 'generation_provider' ] = 'gemini'
	st.session_state[ 'generation_status' ] = 'generating'
	st.session_state[ 'gemini_grounding_error' ] = ''
	st.session_state[ 'gemini_grounding_sources' ] = [ ]
	
	response_buffer: str = ''
	
	if stream:
		if output is None:
			output = st.empty( )
		
		def stream_handler( chunk: str ) -> None:
			"""Render one streamed Gemini response chunk.

			Purpose:
				Appends one text chunk returned by the Gemini wrapper to the current
				response buffer and refreshes the Streamlit output container.

			Args:
				chunk: Text fragment returned by the Gemini streaming response.

			Returns:
				None: This function performs its work through side effects.
			"""
			nonlocal response_buffer
			
			chunk_value: str = str( chunk or '' )
			
			if not chunk_value:
				return
			
			response_buffer += chunk_value
			
			output.markdown( response_buffer + '▌' )
		
		response: str | None = chat.generate_text( prompt=effective_prompt, model=model_value,
			temperature=float( generation_parameters[ 'temperature' ] ),
			top_p=float( generation_parameters[ 'top_p' ] ),
			top_k=int( generation_parameters[ 'top_k' ] ),
			frequency=float( generation_parameters[ 'frequency_penalty' ] ),
			presence=float( generation_parameters[ 'presence_penalty' ] ),
			max_tokens=int( generation_parameters[ 'max_tokens' ] ), tools=[ 'google_search' ],
			tool_choice='AUTO', stream=True, stream_handler=stream_handler, )
		
		response_text: str = str( response or response_buffer or '' ).strip( )
		
		output.markdown( response_text )
		
		st.session_state[ 'generation_status' ] = 'completed'
		
		return response_text
	
	response = chat.generate_text( prompt=effective_prompt, model=model_value,
		temperature=float( generation_parameters[ 'temperature' ] ),
		top_p=float( generation_parameters[ 'top_p' ] ),
		top_k=int( generation_parameters[ 'top_k' ] ),
		frequency=float( generation_parameters[ 'frequency_penalty' ] ),
		presence=float( generation_parameters[ 'presence_penalty' ] ),
		max_tokens=int( generation_parameters[ 'max_tokens' ] ), tools=[ 'google_search' ],
		tool_choice='AUTO', stream=False, )
	
	response_text = str( response or '' ).strip( )
	
	st.session_state[ 'gemini_grounding_sources' ] = (chat.get_grounding_sources( ))
	
	st.session_state[ 'generation_status' ] = 'completed'
	
	return response_text

# -------- Semantic Search Utilities

def decode_embedding_rows( ) -> List[ Tuple[ str, np.ndarray ] ]:
	"""
		Purpose:
		--------
		Read and decode rows from the semantic embeddings table.

		Parameters:
		-----------
		None

		Returns:
		--------
		List[Tuple[str, np.ndarray]]
	"""
	rows_out: List[ Tuple[ str, np.ndarray ] ] = [ ]
	
	with sqlite3.connect( cfg.DB_PATH ) as conn:
		rows = conn.execute( 'SELECT chunk, vector FROM embeddings' ).fetchall( )
	
	for chunk_text_value, vector_blob in rows:
		if not vector_blob:
			continue
		
		vec = np.frombuffer( vector_blob, dtype=np.float32 )
		if vec.size == 0:
			continue
		
		rows_out.append( (str( chunk_text_value or '' ), vec) )
	
	return rows_out

def clear_semantic_index( ) -> None:
	"""
		Purpose:
		--------
		Clear the semantic embeddings table and reset Semantic Search diagnostics.

		Parameters:
		-----------
		None

		Returns:
		--------
		None
	"""
	with sqlite3.connect( cfg.DB_PATH ) as conn:
		conn.execute( 'DELETE FROM embeddings' )
		conn.commit( )
	
	st.session_state[ 'semantic_result_rows' ] = [ ]
	st.session_state[ 'semantic_selected_rows' ] = [ ]
	st.session_state[ 'semantic_index_chunk_count' ] = 0
	st.session_state[ 'semantic_index_dim' ] = 0
	st.session_state[ 'semantic_index_doc_count' ] = 0

def build_semantic_index( uploaded_files: List[ Any ] ) -> Dict[ str, Any ]:
	"""
		Purpose:
		--------
		Build or append a semantic chunk index from uploaded files.

		Parameters:
		-----------
		uploaded_files : List[Any]

		Returns:
		--------
		Dict[str, Any]
	"""
	embedder = load_embedder( )
	if embedder is None:
		return {
				'success': False,
				'message': 'Embedding model unavailable.',
				'doc_count': 0,
				'chunk_count': 0,
				'vector_dim': 0,
		}
	
	chunk_size = int( st.session_state.get( 'semantic_chunk_size', 1200 ) )
	chunk_overlap = int( st.session_state.get( 'semantic_chunk_overlap', 200 ) )
	clear_existing = bool( st.session_state.get( 'semantic_clear_existing', True ) )
	append_existing = bool( st.session_state.get( 'semantic_append_existing', False ) )
	
	if clear_existing and not append_existing:
		clear_semantic_index( )
	
	all_chunks: List[ str ] = [ ]
	doc_names: List[ str ] = [ ]
	
	for f in uploaded_files:
		try:
			file_name = str( getattr( f, 'name', '' ) or '' ).strip( )
			file_bytes = f.getvalue( )
		except Exception:
			continue
		
		if not file_name or not file_bytes:
			continue
		
		text = extract_text_from_bytes( file_bytes=file_bytes, file_name=file_name )
		if not text:
			try:
				text = file_bytes.decode( errors='ignore' )
			except Exception:
				text = ''
		
		if not text:
			continue
		
		chunks = chunk_text( text=text, size=chunk_size, overlap=chunk_overlap )
		if not chunks:
			continue
		
		all_chunks.extend( chunks )
		doc_names.append( file_name )
	
	if len( all_chunks ) == 0:
		return {
				'success': False,
				'message': 'No extractable text was found in the uploaded files.',
				'doc_count': 0,
				'chunk_count': 0,
				'vector_dim': 0,
		}
	
	vecs = embedder.encode( all_chunks, show_progress_bar=False )
	vecs = np.asarray( vecs, dtype=np.float32 )
	
	with sqlite3.connect( cfg.DB_PATH ) as conn:
		for chunk_text_value, vec in zip( all_chunks, vecs ):
			conn.execute( 'INSERT INTO embeddings (chunk, vector) VALUES (?, ?)',
				(chunk_text_value, vec.tobytes( ) ) )
		conn.commit( )
	
	vector_dim = int( vecs.shape[ 1 ] ) if len( vecs.shape ) == 2 else 0
	
	st.session_state[ 'semantic_uploaded_names' ] = doc_names
	st.session_state[ 'semantic_index_doc_count' ] = len( doc_names )
	st.session_state[ 'semantic_index_chunk_count' ] = len( all_chunks )
	st.session_state[ 'semantic_index_dim' ] = vector_dim
	
	return {
			'success': True,
			'message': 'Semantic index built successfully.',
			'doc_count': len( doc_names ),
			'chunk_count': len( all_chunks ),
			'vector_dim': vector_dim,
	}

def query_semantic_index( query_text: str ) -> List[ Dict[ str, Any ] ]:
	"""
		Purpose:
		--------
		Query the semantic index and return ranked chunk results.

		Parameters:
		-----------
		query_text : str

		Returns:
		--------
		List[Dict[str, Any]]
	"""
	if not query_text or not query_text.strip( ):
		return [ ]
	
	embedder = load_embedder( )
	if embedder is None:
		return [ ]
	
	top_k = int( st.session_state.get( 'semantic_top_k', 8 ) )
	min_similarity = float( st.session_state.get( 'semantic_min_similarity', 0.0 ) )
	
	rows = decode_embedding_rows( )
	if not rows:
		return [ ]
	
	q = embedder.encode( [ query_text.strip( ) ], show_progress_bar=False )[ 0 ]
	q = np.asarray( q, dtype=np.float32 )
	
	scored_rows: List[ Dict[ str, Any ] ] = [ ]
	for idx, (chunk_text_value, vec) in enumerate( rows, start=1 ):
		score = cosine_similarity( q, vec )
		if score < min_similarity:
			continue
		
		scored_rows.append( {
					'Selected': False,
					'Rank': idx,
					'Score': float( score ),
					'Chunk': chunk_text_value,
					'Length': len( chunk_text_value ),
			} )
	
	scored_rows.sort( key=lambda r: r[ 'Score' ], reverse=True )
	scored_rows = scored_rows[ :top_k ]
	
	if bool( st.session_state.get( 'semantic_group_by_document', False ) ):
		# Current embeddings table stores chunk text but not document name.
		# Preserve current schema and grouping behavior as a no-op until schema expansion.
		pass
	
	st.session_state[ 'semantic_last_query' ] = query_text.strip( )
	st.session_state[ 'semantic_result_rows' ] = scored_rows
	return scored_rows

def build_semantic_context_from_selection( ) -> str:
	"""
		Purpose:
		--------
		Build a semantic-context text block from selected search rows.

		Parameters:
		-----------
		None

		Returns:
		--------
		str
	"""
	selected_rows = st.session_state.get( 'semantic_selected_rows', [ ] )
	if not isinstance( selected_rows, list ) or len( selected_rows ) == 0:
		return ''
	
	context_parts: List[ str ] = [ ]
	for idx, row in enumerate( selected_rows, start=1 ):
		chunk_text_value = str( row.get( 'Chunk', '' ) or '' ).strip( )
		score_value = row.get( 'Score', '' )
		if not chunk_text_value:
			continue
		
		context_parts.append(
			f'[Semantic Chunk {idx} | Score: {score_value}]\n{chunk_text_value}'
		)
	
	return '\n\n'.join( context_parts ).strip( )

def send_selected_semantic_chunks_to_text_generation( ) -> None:
	"""
		Purpose:
		--------
		Push selected semantic chunks into the shared basic document context buffer.

		Parameters:
		-----------
		None

		Returns:
		--------
		None
	"""
	context_text = build_semantic_context_from_selection( )
	if not context_text:
		return
	
	existing_docs = st.session_state.get( 'basic_docs', [ ] )
	if not isinstance( existing_docs, list ):
		existing_docs = [ ]
	
	existing_docs.append( context_text )
	st.session_state[ 'basic_docs' ] = existing_docs
	st.session_state[ 'use_semantic' ] = True

def send_selected_semantic_chunks_to_doc_qna( ) -> None:
	"""
		Purpose:
		--------
		Push selected semantic chunks into the shared document context buffer used by prompts.

		Parameters:
		-----------
		None

		Returns:
		--------
		None
	"""
	context_text = build_semantic_context_from_selection( )
	if not context_text:
		return
	
	buffer_rows = st.session_state.get( 'semantic_context_buffer', [ ] )
	if not isinstance( buffer_rows, list ):
		buffer_rows = [ ]
	
	buffer_rows.append( context_text )
	st.session_state[ 'semantic_context_buffer' ] = buffer_rows

def extract_selected_semantic_rows( edited_rows: List[ Dict[ str, Any ] ] ) -> List[ Dict[ str, Any ] ]:
	"""
		Purpose:
		--------
		Extract selected semantic rows from a data_editor result payload.

		Parameters:
		-----------
		edited_rows : List[Dict[str, Any]]

		Returns:
		--------
		List[Dict[str, Any]]
	"""
	selected: List[ Dict[ str, Any ] ] = [ ]
	if not isinstance( edited_rows, list ):
		return selected
	
	for row in edited_rows:
		if isinstance( row, dict ) and bool( row.get( 'Selected', False ) ):
			selected.append( row )
	
	return selected

# -------- CHAT/TEXT UTILITIES --------------------

def normalize_text( text: str ) -> str:
	"""
		
		Purpose
		-------
		Normalize text by:
			• Converting to lowercase
			• Removing punctuation except sentence delimiters (. ! ?)
			• Ensuring clean sentence boundary spacing
			• Collapsing whitespace
	
		Parameters
		----------
		text: str
	
		Returns
		-------
		str
		
	"""
	if not text:
		return ""
	
	# Lowercase
	text = text.lower( )
	
	# Remove punctuation except . ! ?
	text = re.sub( r"[^\w\s\.\!\?]", "", text )
	
	# Ensure single space after sentence delimiters
	text = re.sub( r"([.!?])\s*", r"\1 ", text )
	
	# Normalize whitespace
	text = re.sub( r"\s+", " ", text ).strip( )
	
	return text

def chunk_text( text: str, size: int | None = None, overlap: int | None = None ) -> List[ str ]:
	"""
		Purpose:
		--------
		Split text into overlapping chunks using session-state defaults when explicit values
		are not provided.

		Parameters:
		-----------
		text : str
		size : int | None
		overlap : int | None

		Returns:
		--------
		List[str]
	"""
	if not text:
		return [ ]
	
	chunk_size = int(
		size if size is not None else st.session_state.get( 'retrieval_chunk_size', 1200 )
	)
	chunk_overlap = int(
		overlap if overlap is not None else st.session_state.get( 'retrieval_chunk_overlap', 200 )
	)
	
	if chunk_size <= 0:
		chunk_size = 1200
	
	if chunk_overlap < 0:
		chunk_overlap = 0
	
	if chunk_overlap >= chunk_size:
		chunk_overlap = max( 0, chunk_size // 4 )
	
	chunks: List[ str ] = [ ]
	i = 0
	step = max( 1, chunk_size - chunk_overlap )
	
	while i < len( text ):
		chunk = text[ i:i + chunk_size ]
		if chunk and chunk.strip( ):
			chunks.append( chunk )
		i += step
	
	return chunks

def convert_xml( text: str ) -> str:
	"""
		
			Purpose:
			_________
			Convert XML-delimited prompt text into Markdown by treating XML-like
			tags as section delimiters, not as strict XML.
	
			Parameters:
			-----------
			text (str) - Prompt text containing XML-like opening and closing tags.
	
			Returns:
			---------
			Markdown-formatted text using level-2 headings (##).
	"""
	markdown_blocks: List[ str ] = [ ]
	for match in cfg.XML_BLOCK_PATTERN.finditer( text ):
		raw_tag: str = match.group( "tag" )
		body: str = match.group( "body" ).strip( )
		
		# Humanize tag name for Markdown heading
		heading: str = raw_tag.replace( "_", " " ).replace( "-", " " ).title( )
		markdown_blocks.append( f"## {heading}" )
		if body:
			markdown_blocks.append( body )
	return "\n\n".join( markdown_blocks )

def convert_markdown( text: Any ) -> str:
	"""
		Purpose:
		--------
		Convert between Markdown headings and simple XML-like heading tags.
	
		Behavior:
		---------
		Auto-detects direction:
		  - If <h1>...</h1> / <h2>...</h2> ... exist, converts to Markdown (# / ## / ###).
		  - Otherwise converts Markdown headings (# / ## / ###) to <hN>...</hN> tags.
	
		Parameters:
		-----------
		text : Any
			Source text. Non-string values return "".
	
		Returns:
		--------
		str
			Converted text.
	"""
	if not isinstance( text, str ) or not text.strip( ):
		return ""
	
	# Normalize newlines
	src = text.replace( "\r\n", "\n" ).replace( "\r", "\n" )	
	htag_pattern = re.compile( r"<h([1-6])>(.*?)</h\1>", flags=re.IGNORECASE | re.DOTALL )
	md_heading_pattern = re.compile( r"^(#{1,6})[ \t]+(.+?)[ \t]*$", flags=re.MULTILINE )
	
	# ------------------------------------------------------------------
	# Direction detection
	# ------------------------------------------------------------------
	contains_htags = bool( htag_pattern.search( src ) )
	
	# ------------------------------------------------------------------
	# XML-like heading tags -> Markdown headings
	# ------------------------------------------------------------------
	if contains_htags:
		def _htag_to_md( match: re.Match ) -> str:
			level = int( match.group( 1 ) )
			content = match.group( 2 ).strip( )
			
			# Preserve inner newlines safely by collapsing interior whitespace
			# while keeping content readable.
			content = re.sub( r"[ \t]+\n", "\n", content )
			content = re.sub( r"\n[ \t]+", "\n", content )
			
			return f"{'#' * level} {content}"
		
		out = htag_pattern.sub( _htag_to_md, src )
		return out.strip( )
	
	# ------------------------------------------------------------------
	# Markdown headings -> XML-like heading tags
	# ------------------------------------------------------------------
	def _md_to_htag( match: re.Match ) -> str:
		hashes = match.group( 1 )
		content = match.group( 2 ).strip( )
		level = len( hashes )
		return f"<h{level}>{content}</h{level}>"
	
	out = md_heading_pattern.sub( _md_to_htag, src )
	return out.strip( )

def inject_response_css( ) -> None:
	"""
	
		Purpose:
		_________
		Set the the format via css.
		
	"""
	st.markdown(
		"""
		<style>
		/* Chat message text */
		.stChatMessage p {
			color: rgb(220, 220, 220);
			font-size: 1rem;
			line-height: 1.6;
		}

		/* Headings inside chat responses */
		.stChatMessage h1 {
			color: rgb(0, 120, 252); /* DoD Blue */
			font-size: 1.6rem;
		}

		.stChatMessage h2 {
			color: rgb(0, 120, 252);
			font-size: 1.35rem;
		}

		.stChatMessage h3 {
			color: rgb(0, 120, 252);
			font-size: 1.15rem;
		}
		
		.stChatMessage a {
			color: rgb(0, 120, 252); /* DoD Blue */
			text-decoration: underline;
		}
		
		.stChatMessage a:hover {
			color: rgb(80, 160, 255);
		}

		</style>
		""", unsafe_allow_html=True )

def style_subheaders( ) -> None:
	"""
	
		Purpose:
		_________
		Sets the style of subheaders in the main UI
		
	"""
	st.markdown(
		"""
		<style>
		div[data-testid="stMarkdownContainer"] h2,
		div[data-testid="stMarkdownContainer"] h3,
		div[data-testid="stChatMessage"] div[data-testid="stMarkdownContainer"] h2,
		div[data-testid="stChatMessage"] div[data-testid="stMarkdownContainer"] h3 {
			color: rgb(0, 120, 252) !important;
		}
		</style>
		""",
		unsafe_allow_html=True, )

def save_message( role: str, content: str ) -> None:
	with sqlite3.connect( cfg.DB_PATH ) as conn:
		conn.execute( 'INSERT INTO chat_history (role, content) VALUES (?, ?)', (role, content) )

def load_history( ) -> List[ Tuple[ str, str ] ]:
	with sqlite3.connect( cfg.DB_PATH ) as conn:
		return conn.execute( 'SELECT role, content FROM chat_history ORDER BY id' ).fetchall( )

def clear_history( ) -> None:
	with sqlite3.connect( cfg.DB_PATH ) as conn:
		conn.execute( "DELETE FROM chat_history" )

# -------- PROMPT ENGINEERING UTILITIES ----------------

def fetch_prompt_categories( ) -> List[ str ]:
	"""Prompt category retrieval.

	Purpose:
	    Retrieves the distinct non-empty prompt categories stored in the Prompts table.

	Args:
	    None.

	Returns:
	    List[str]: Sorted prompt category values.
	"""
	with sqlite3.connect( cfg.DB_PATH ) as conn:
		rows: List[ Tuple[ Any, ... ] ] = conn.execute( """
                                                        SELECT DISTINCT TRIM("Category") AS
	                                                                        "Category"
                                                        FROM "Prompts"
                                                        WHERE "Category" IS NOT NULL
                                                          AND TRIM("Category") <> ''
                                                        ORDER BY TRIM("Category") ASC;
		                                                """ ).fetchall( )
	
	return [ str( row[ 0 ] ).strip( ) for row in rows if
		row and row[ 0 ] is not None and str( row[ 0 ] ).strip( ) ]

def fetch_prompt_options( category: str | None = None ) -> List[ Dict[ str, Any ] ]:
	"""Prompt option retrieval.

	Purpose:
	    Retrieves prompt records for use in ID-backed Streamlit selection controls.
	    Results may be restricted to one stored prompt category.

	Args:
	    category (str | None): Optional category used to filter prompt records.

	Returns:
	    List[Dict[str, Any]]: Prompt option records containing ID, Caption, Name,
	    Category, and Text.
	"""
	category_value: str = str( category or '' ).strip( )
	query: str = """
                 SELECT "ID",
                        "Caption",
                        "Name",
                        "Category",
                        "Text"
                 FROM "Prompts"
                 WHERE "Text" IS NOT NULL
                   AND TRIM("Text") <> '' \
	             """
	
	params: List[ Any ] = [ ]
	if category_value:
		query += """
			AND "Category" = ?
		"""
		params.append( category_value )
	
	query += """
		ORDER BY
			TRIM("Caption") ASC,
			"ID" ASC;
	"""
	
	with sqlite3.connect( cfg.DB_PATH ) as conn:
		conn.row_factory = sqlite3.Row
		rows: List[ sqlite3.Row ] = conn.execute( query, params, ).fetchall( )
	
	return [ { 'ID': int( row[ 'ID' ] ), 'Caption': str( row[ 'Caption' ] or '' ).strip( ),
		'Name': str( row[ 'Name' ] or '' ).strip( ),
		'Category': str( row[ 'Category' ] or '' ).strip( ), 'Text': str( row[ 'Text' ] or '' ), }
		for row in rows ]

def fetch_prompt_by_id( prompt_id: int ) -> Dict[ str, Any ] | None:
	"""Prompt record retrieval.

	Purpose:
	    Retrieves one prompt record by its integer primary key.

	Args:
	    prompt_id (int): Primary-key value of the prompt record.

	Returns:
	    Dict[str, Any] | None: Prompt record when found; otherwise None.
	"""
	if int( prompt_id ) <= 0:
		return None
	
	with sqlite3.connect( cfg.DB_PATH ) as conn:
		conn.row_factory = sqlite3.Row
		row: sqlite3.Row | None = conn.execute( """
                                                SELECT "ID",
                                                       "Caption",
                                                       "Name",
                                                       "Category",
                                                       "Text"
                                                FROM "Prompts"
                                                WHERE "ID" = ?;
		                                        """, (int( prompt_id ),), ).fetchone( )
	
	if row is None:
		return None
	
	return { 'ID': int( row[ 'ID' ] ), 'Caption': str( row[ 'Caption' ] or '' ).strip( ),
		'Name': str( row[ 'Name' ] or '' ).strip( ),
		'Category': str( row[ 'Category' ] or '' ).strip( ), 'Text': str( row[ 'Text' ] or '' ), }

def fetch_prompts_df( search_text: str = '', category: str = '', sort_column: str = 'ID',
	sort_direction: str = 'ASC', limit: int | None = None, offset: int = 0, ) -> pd.DataFrame:
	"""Prompt table retrieval.

	Purpose:
	    Retrieves prompt records for Prompt Engineering mode using validated search,
	    category, sorting, paging, and offset values.

	Args:
	    search_text (str): Search text applied to Caption, Name, Category, and Text.
	    category (str): Exact stored category filter.
	    sort_column (str): Validated database column used for sorting.
	    sort_direction (str): Sort direction using ASC or DESC.
	    limit (int | None): Optional maximum number of records to return.
	    offset (int): Number of matching records to skip.

	Returns:
	    pd.DataFrame: Prompt records using the Prompts table schema.
	"""
	valid_sort_columns: List[ str ] = [ 'ID', 'Caption', 'Name', 'Category', 'Text', ]
	sort_column_value: str = (sort_column if sort_column in valid_sort_columns else 'ID')
	sort_direction_value: str = ( 'DESC' if str( sort_direction or '' ).upper( ) == 'DESC' else 'ASC')
	search_value: str = str( search_text or '' ).strip( )
	category_value: str = str( category or '' ).strip( )
	where_clauses: List[ str ] = [ ]
	params: List[ Any ] = [ ]
	if search_value:
		where_clauses.append( """
			(
				"Caption" LIKE ?
				OR "Name" LIKE ?
				OR "Category" LIKE ?
				OR "Text" LIKE ?
			)
			""" )
		
		search_pattern: str = f'%{search_value}%'
		params.extend( [ search_pattern, search_pattern, search_pattern, search_pattern, ] )
	
	if category_value:
		where_clauses.append( '"Category" = ?' )
		params.append( category_value )
	
	where_sql: str = ''
	if where_clauses:
		where_sql = 'WHERE ' + ' AND '.join( where_clauses )
	
	query: str = f"""
		SELECT
			"ID",
			"Caption",
			"Name",
			"Category",
			"Text"
		FROM "Prompts"
		{where_sql}
		ORDER BY
			"{sort_column_value}" {sort_direction_value}
	"""
	
	if limit is not None:
		limit_value: int = max( 1, int( limit ) )
		offset_value: int = max( 0, int( offset ) )
		
		query += """
			LIMIT ?
			OFFSET ?
		"""
		
		params.extend( [ limit_value, offset_value, ] )
	
	with sqlite3.connect( cfg.DB_PATH ) as conn:
		df_prompts: pd.DataFrame = pd.read_sql_query( query, conn, params=params, )
	
	return df_prompts

def count_prompts( search_text: str = '', category: str = '', ) -> int:
	"""Prompt record counting.

	Purpose:
	    Counts prompt records using the same search and category filters applied by
	    Prompt Engineering mode.

	Args:
	    search_text (str): Search text applied to Caption, Name, Category, and Text.
	    category (str): Exact stored category filter.

	Returns:
	    int: Number of matching prompt records.
	"""
	search_value: str = str( search_text or '' ).strip( )
	category_value: str = str( category or '' ).strip( )
	
	where_clauses: List[ str ] = [ ]
	params: List[ Any ] = [ ]
	
	if search_value:
		where_clauses.append( """
			(
				"Caption" LIKE ?
				OR "Name" LIKE ?
				OR "Category" LIKE ?
				OR "Text" LIKE ?
			)
			""" )
		
		search_pattern: str = f'%{search_value}%'
		params.extend( [ search_pattern, search_pattern, search_pattern, search_pattern, ] )
	
	if category_value:
		where_clauses.append( '"Category" = ?' )
		params.append( category_value )
	
	where_sql: str = ''
	
	if where_clauses:
		where_sql = 'WHERE ' + ' AND '.join( where_clauses )
	
	with sqlite3.connect( cfg.DB_PATH ) as conn:
		row: Tuple[ Any, ... ] | None = conn.execute( f"""
			SELECT
				COUNT(*)
			FROM "Prompts"
			{where_sql};
			""", params, ).fetchone( )
	
	return int( row[ 0 ] ) if row else 0

def validate_prompt_record( data: Dict[ str, Any ] ) -> Dict[ str, str ]:
	"""Prompt record validation.

	Purpose:
	    Validates and normalizes the editable values required to create or update a
	    Prompts table record.

	Args:
	    data (Dict[str, Any]): Prompt record values using Caption, Name, Category,
	    and Text keys.

	Returns:
	    Dict[str, str]: Normalized prompt values.

	Raises:
	    ValueError: Raised when a required prompt value is empty or exceeds the
	    declared schema length.
	"""
	caption: str = str( data.get( 'Caption', '' ) or '' ).strip( )
	name: str = str( data.get( 'Name', '' ) or '' ).strip( )
	category: str = str( data.get( 'Category', '' ) or '' ).strip( )
	text: str = str( data.get( 'Text', '' ) or '' ).strip( )
	
	if not caption:
		raise ValueError( 'Caption is required.' )
	
	if not name:
		raise ValueError( 'Name is required.' )
	
	if not category:
		raise ValueError( 'Category is required.' )
	
	if not text:
		raise ValueError( 'Text is required.' )
	
	if len( caption ) > 80:
		raise ValueError( 'Caption cannot exceed 80 characters.' )
	
	if len( name ) > 80:
		raise ValueError( 'Name cannot exceed 80 characters.' )
	
	if len( category ) > 80:
		raise ValueError( 'Category cannot exceed 80 characters.' )
	
	if len( text ) > 2048:
		raise ValueError( 'Text cannot exceed 2048 characters.' )
	
	return { 'Caption': caption, 'Name': name, 'Category': category, 'Text': text, }

def insert_prompt( data: Dict[ str, Any ] ) -> int:
	"""Prompt record insertion.

	Purpose:
	    Creates one validated prompt record and returns its generated primary key.

	Args:
	    data (Dict[str, Any]): Prompt record values using Caption, Name, Category,
	    and Text keys.

	Returns:
	    int: Generated Prompts.ID value.
	"""
	validated_data: Dict[ str, str ] = validate_prompt_record( data )
	
	with sqlite3.connect( cfg.DB_PATH ) as conn:
		cursor: sqlite3.Cursor = conn.execute( """
                                               INSERT INTO "Prompts"
                                               ("Caption",
                                                "Name",
                                                "Category",
                                                "Text")
                                               VALUES (?,
                                                       ?,
                                                       ?,
                                                       ?);
		                                       """,
			(validated_data[ 'Caption' ], validated_data[ 'Name' ], validated_data[ 'Category' ],
				validated_data[ 'Text' ],), )
		
		conn.commit( )
		
		prompt_id: int = int( cursor.lastrowid )
	
	return prompt_id

def update_prompt( prompt_id: int, data: Dict[ str, Any ] ) -> None:
	"""Prompt record update.

	Purpose:
	    Updates the editable values of one prompt record without modifying its primary key.

	Args:
	    prompt_id (int): Primary-key value of the prompt record.
	    data (Dict[str, Any]): Prompt record values using Caption, Name, Category,
	    and Text keys.

	Returns:
	    None: This function performs its work through side effects and does not return
	    a value.

	Raises:
	    ValueError: Raised when the prompt identifier is invalid or the target record
	    does not exist.
	"""
	prompt_id_value: int = int( prompt_id )
	
	if prompt_id_value <= 0:
		raise ValueError( 'A valid prompt ID is required.' )
	
	validated_data: Dict[ str, str ] = validate_prompt_record( data )
	
	with sqlite3.connect( cfg.DB_PATH ) as conn:
		cursor: sqlite3.Cursor = conn.execute( """
                                               UPDATE "Prompts"
                                               SET "Caption"  = ?,
                                                   "Name"     = ?,
                                                   "Category" = ?,
                                                   "Text"     = ?
                                               WHERE "ID" = ?;
		                                       """,
			(validated_data[ 'Caption' ], validated_data[ 'Name' ], validated_data[ 'Category' ],
				validated_data[ 'Text' ], prompt_id_value,), )
		
		if cursor.rowcount == 0:
			raise ValueError( f'Prompt ID {prompt_id_value} was not found.' )
		
		conn.commit( )

def delete_prompt( prompt_id: int ) -> None:
	"""Prompt record deletion.

	Purpose:
	    Deletes one prompt record by its integer primary key.

	Args:
	    prompt_id (int): Primary-key value of the prompt record.

	Returns:
	    None: This function performs its work through side effects and does not return
	    a value.

	Raises:
	    ValueError: Raised when the prompt identifier is invalid or the target record
	    does not exist.
	"""
	prompt_id_value: int = int( prompt_id )
	
	if prompt_id_value <= 0:
		raise ValueError( 'A valid prompt ID is required.' )
	
	with sqlite3.connect( cfg.DB_PATH ) as conn:
		cursor: sqlite3.Cursor = conn.execute( """
                                               DELETE
                                               FROM "Prompts"
                                               WHERE "ID" = ?;
		                                       """, (prompt_id_value,), )
		
		if cursor.rowcount == 0:
			raise ValueError( f'Prompt ID {prompt_id_value} was not found.' )
		
		conn.commit( )

def format_prompt_option( prompt_id: int, prompt_options: List[ Dict[ str, Any ] ], ) -> str:
	"""Prompt option formatting.

	Purpose:
	    Formats a prompt identifier using the stored Caption value for display in
	    Streamlit selection controls.

	Args:
	    prompt_id (int): Selected prompt identifier.
	    prompt_options (List[Dict[str, Any]]): Prompt option records available to the
	    selection control.

	Returns:
	    str: User-facing prompt caption.
	"""
	prompt_id_value: int = int( prompt_id )
	prompt_record: Dict[ str, Any ] | None = next( (option for option in prompt_options if
	int( option.get( 'ID', 0 ) or 0 ) == prompt_id_value), None, )
	
	if prompt_record is None:
		return str( prompt_id_value )
	
	caption: str = str( prompt_record.get( 'Caption', '', ) or '' ).strip( )
	return (caption if caption else f'Prompt {prompt_id_value}')

# ----- Instruction Selection Utilities --------

def get_available_prompt_categories( allowed_categories: List[ str ], ) -> List[ str ]:
	"""Available prompt category retrieval.

	Purpose:
	    Returns stored prompt categories that are permitted for the current application
	    mode. The database remains authoritative for category availability while the
	    supplied allowlist controls mode eligibility.

	Args:
	    allowed_categories (List[str]): Prompt categories permitted for the active mode.

	Returns:
	    List[str]: Stored prompt categories that are present in the supplied allowlist.
	"""
	stored_categories: List[ str ] = fetch_prompt_categories( )
	
	return [ category for category in allowed_categories if category in stored_categories ]

def get_prompt_ids_for_category( category: str, ) -> Tuple[ List[ int ], List[ Dict[ str, Any ] ] ]:
	"""Category prompt option retrieval.

	Purpose:
	    Retrieves prompt records for one category and separates their integer identifiers
	    for use by an ID-backed Streamlit selectbox.

	Args:
	    category (str): Stored prompt category used to filter prompt records.

	Returns:
	    Tuple[List[int], List[Dict[str, Any]]]: Prompt identifiers and their corresponding
	    prompt records.
	"""
	category_value: str = str( category or '' ).strip( )
	
	if not category_value:
		return [ ], [ ]
	
	prompt_options: List[ Dict[ str, Any ] ] = fetch_prompt_options( category=category_value, )
	
	prompt_ids: List[ int ] = [ int( option[ 'ID' ] ) for option in prompt_options if
		int( option.get( 'ID', 0 ) or 0 ) > 0 ]
	
	return prompt_ids, prompt_options

def initialize_system_instruction_state( category_key: str, prompt_id_key: str,
	allowed_categories: List[ str ], ) -> None:
	"""System instruction state initialization.

	Purpose:
	    Initializes mode-specific category and prompt-selection state before Streamlit
	    controls read those values.

	Args:
	    category_key (str): Session-state key used by the category selector.
	    prompt_id_key (str): Session-state key used by the prompt-ID selector.
	    allowed_categories (List[str]): Prompt categories permitted for the active mode.

	Returns:
	    None: This function performs its work through side effects and does not return
	    a value.
	"""
	available_categories: List[ str ] = get_available_prompt_categories(
		allowed_categories=allowed_categories, )
	
	if category_key not in st.session_state:
		st.session_state[ category_key ] = (
			available_categories[ 0 ] if available_categories else '')
	
	current_category: str = str( st.session_state.get( category_key, '' ) or '' ).strip( )
	
	if current_category not in available_categories:
		st.session_state[ category_key ] = (
			available_categories[ 0 ] if available_categories else '')
	
	if prompt_id_key not in st.session_state:
		st.session_state[ prompt_id_key ] = None

def clear_active_prompt_metadata( ) -> None:
	"""Active prompt metadata reset.

	Purpose:
	    Clears shared metadata describing the currently loaded prompt without changing
	    the editable System Instructions text.

	Args:
	    None.

	Returns:
	    None: This function performs its work through side effects and does not return
	    a value.
	"""
	st.session_state[ 'selected_prompt_id' ] = None
	st.session_state[ 'active_prompt_caption' ] = ''
	st.session_state[ 'active_prompt_name' ] = ''

def clear_system_instruction_selection( prompt_id_key: str, clear_text: bool = False, ) -> None:
	"""System instruction selection reset.

	Purpose:
	    Clears one mode-specific prompt selection and its shared prompt metadata. The
	    editable System Instructions text is cleared only when explicitly requested.

	Args:
	    prompt_id_key (str): Session-state key used by the prompt-ID selector.
	    clear_text (bool): Indicates whether the editable System Instructions text should
	    also be cleared.

	Returns:
	    None: This function performs its work through side effects and does not return
	    a value.
	"""
	st.session_state[ prompt_id_key ] = None
	clear_active_prompt_metadata( )
	
	if clear_text:
		st.session_state[ 'system_instructions' ] = ''

def change_system_instruction_category( category_key: str, prompt_id_key: str, ) -> None:
	"""System instruction category change handler.

	Purpose:
	    Clears a stale prompt selection when the user changes the selected category.
	    Manually edited System Instructions remain unchanged.

	Args:
	    category_key (str): Session-state key used by the category selector.
	    prompt_id_key (str): Session-state key used by the prompt-ID selector.

	Returns:
	    None: This function performs its work through side effects and does not return
	    a value.
	"""
	_ = str( st.session_state.get( category_key, '' ) or '' ).strip( )
	
	clear_system_instruction_selection( prompt_id_key=prompt_id_key, clear_text=False, )

def load_selected_prompt_into_system_instructions( prompt_id_key: str, ) -> None:
	"""Selected prompt loading.

	Purpose:
	    Loads the Text value of the selected prompt record into the shared editable
	    System Instructions state and records the selected prompt metadata.

	Args:
	    prompt_id_key (str): Session-state key containing the selected prompt ID.

	Returns:
	    None: This function performs its work through side effects and does not return
	    a value.
	"""
	selected_value: Any = st.session_state.get( prompt_id_key )
	
	if selected_value is None or selected_value == '':
		clear_active_prompt_metadata( )
		return
	
	try:
		prompt_id: int = int( selected_value )
	except (TypeError, ValueError):
		clear_system_instruction_selection( prompt_id_key=prompt_id_key, clear_text=False, )
		return
	
	prompt_record: Dict[ str, Any ] | None = fetch_prompt_by_id( prompt_id=prompt_id, )
	if prompt_record is None:
		clear_system_instruction_selection( prompt_id_key=prompt_id_key, clear_text=False, )
		return
	
	st.session_state[ 'system_instructions' ] = str( prompt_record.get( 'Text', '' ) or '' )
	st.session_state[ 'selected_prompt_id' ] = int( prompt_record[ 'ID' ] )
	st.session_state[ 'active_prompt_caption' ] = str(
		prompt_record.get( 'Caption', '' ) or '' ).strip( )
	
	st.session_state[ 'active_prompt_name' ] = str( prompt_record.get( 'Name', '' ) or ''
	).strip( )

def reset_system_instruction_controls( category_key: str, prompt_id_key: str,
	allowed_categories: List[ str ], clear_text: bool = True, ) -> None:
	"""System instruction control reset.

	Purpose:
	    Restores one System Instructions selector group to its initial category and clears
	    its prompt selection, metadata, and optionally the editable instruction text.

	Args:
	    category_key (str): Session-state key used by the category selector.
	    prompt_id_key (str): Session-state key used by the prompt-ID selector.
	    allowed_categories (List[str]): Prompt categories permitted for the active mode.
	    clear_text (bool): Indicates whether the editable System Instructions text should
	    also be cleared.

	Returns:
	    None: This function performs its work through side effects and does not return
	    a value.
	"""
	available_categories: List[ str ] = get_available_prompt_categories(
		allowed_categories=allowed_categories, )
	
	st.session_state[ category_key ] = (available_categories[ 0 ] if available_categories else '')
	st.session_state[ prompt_id_key ] = None
	clear_active_prompt_metadata( )
	if clear_text:
		st.session_state[ 'system_instructions' ] = ''

def convert_system_instruction_text( ) -> None:
	"""System instruction format conversion.

	Purpose:
	    Converts the shared editable System Instructions text between supported XML-like
	    and Markdown heading formats.

	Args:
	    None.

	Returns:
	    None: This function performs its work through side effects and does not return
	    a value.
	"""
	text: Any = st.session_state.get(
		'system_instructions',
		'',
	)

	if not isinstance( text, str ) or not text.strip( ):
		return

	source_text: str = text.strip( )

	if cfg.XML_BLOCK_PATTERN.search( source_text ):
		converted_text: str = convert_xml( source_text )
	else:
		converted_text = convert_markdown( source_text )

	st.session_state[ 'system_instructions' ] = converted_text

def apply_text_generation_preset( ) -> None:
	"""Text Generation preset application.

	Purpose:
	    Applies the selected Text Generation task preset to the shared editable System
	    Instructions state.

	Args:
	    None.

	Returns:
	    None: This function performs its work through side effects and does not return
	    a value.
	"""
	task_preset: str = str( st.session_state.get( 'task_preset', 'Chat', ) or 'Chat' ).strip( )
	preset_map: Dict[ str, str ] = {
		'Chat': 'You are a helpful local assistant. Be accurate, practical, and concise.',
		'Reasoning': 'Solve the task carefully, verify the material conclusion, and provide a '
		             'clear final answer.',
		'Coding': 'Produce correct, editor-ready code and explain only what is necessary for '
		          'implementation.',
		'Translation': 'Translate faithfully while preserving meaning, tone, and structure.',
		'Summarization': 'Summarize faithfully and preserve key facts, names, dates, '
		                 'and conclusions.',
		'Extraction': 'Extract only supported facts and do not invent missing values.', }
	
	st.session_state[ 'system_instructions' ] = preset_map.get( task_preset, preset_map[ 'Chat'
	], )
	
	clear_active_prompt_metadata( )

def get_effective_system_instructions( ) -> str:
	"""
		Purpose:
		--------
		Return the authoritative system instructions text from session state.

		Parameters:
		-----------
		None

		Returns:
		--------
		str
	"""
	text = st.session_state.get( 'system_instructions', '' )
	return str( text ).strip( ) if text is not None else ''

def build_task_instruction_block( ) -> str:
	"""
		Purpose:
		--------
		Build a task-specific instruction block for Text Generation mode.

		Parameters:
		-----------
		None

		Returns:
		--------
		str
	"""
	task_preset = str( st.session_state.get( 'task_preset', 'Chat' ) or 'Chat' ).strip( )
	response_format = str(
		st.session_state.get( 'response_format', 'Markdown' ) or 'Markdown'
	).strip( )
	reasoning_depth = str(
		st.session_state.get( 'reasoning_depth', 'Medium' ) or 'Medium'
	).strip( )
	answer_only = bool( st.session_state.get( 'answer_only', False ) )
	use_self_check = bool( st.session_state.get( 'use_self_check', False ) )
	deterministic_reasoning = bool( st.session_state.get( 'deterministic_reasoning', False ) )
	coding_language = str(
		st.session_state.get( 'coding_language', 'Python' ) or 'Python'
	).strip( )
	coding_task = str( st.session_state.get( 'coding_task', 'Generate' ) or 'Generate' ).strip( )
	coding_include_comments = bool( st.session_state.get( 'coding_include_comments', True ) )
	coding_editor_format = bool( st.session_state.get( 'coding_editor_format', True ) )
	coding_fenced_output = bool( st.session_state.get( 'coding_fenced_output', True ) )
	translation_target_language = str(
		st.session_state.get( 'translation_target_language', 'English' ) or 'English'
	).strip( )
	
	lines: List[ str ] = [ ]
	lines.append( 'Task Preset:' )
	lines.append( f'- Active Task: {task_preset}' )
	lines.append( f'- Response Format: {response_format}' )
	
	if task_preset == 'Reasoning':
		lines.append( f'- Reasoning Depth: {reasoning_depth}' )
		lines.append(
			'- Use a careful analytical process internally and return a clear final answer.'
		)
		if answer_only:
			lines.append( '- Return the final answer without extra prefatory narration.' )
		if use_self_check:
			lines.append( '- Verify the conclusion against the prompt before answering.' )
		if deterministic_reasoning:
			lines.append( '- Prefer stable, conservative reasoning over creative variation.' )
	
	elif task_preset == 'Coding':
		lines.append( f'- Code Language: {coding_language}' )
		lines.append( f'- Coding Task: {coding_task}' )
		if coding_include_comments:
			lines.append(
				'- Include documentation comments and useful inline comments when appropriate.' )
		else:
			lines.append( '- Minimize comments unless required for clarity.' )
		if coding_editor_format:
			lines.append(
				'- Format the output as editor-ready source code, not as explanatory pseudo-code.'
			)
		if coding_fenced_output:
			lines.append(
				'- Return code inside fenced markdown code blocks when code is produced.' )
		else:
			lines.append(
				'- Return raw code without fenced markdown blocks when code is produced.' )
	
	elif task_preset == 'Translation':
		lines.append( f'- Translate the user content into {translation_target_language}.' )
		lines.append( '- Preserve original meaning, tone, and structure where practical.' )
	
	elif task_preset == 'Summarization':
		lines.append( '- Summarize the user content clearly and faithfully.' )
		lines.append( '- Preserve key facts, names, dates, and conclusions.' )
	
	elif task_preset == 'Extraction':
		lines.append( '- Extract the requested facts faithfully and do not invent missing values.' )
		if response_format == 'JSON':
			lines.append( '- Return valid JSON only.' )
	
	else:
		lines.append( '- Respond as a general-purpose assistant.' )
	
	return '\n'.join( lines ).strip( )

def build_effective_prompt_preview( user_input: str ) -> str:
	"""
		Purpose:
		--------
		Build a readable preview of the effective prompt content used for generation.

		Parameters:
		-----------
		user_input : str

		Returns:
		--------
		str
	"""
	system_instructions = get_effective_system_instructions( )
	task_block = build_task_instruction_block( )
	preview_parts: List[ str ] = [ ]
	
	if system_instructions:
		preview_parts.append( '[System Instructions]' )
		preview_parts.append( system_instructions )
	
	if task_block:
		preview_parts.append( '[Task Instructions]' )
		preview_parts.append( task_block )
	
	preview_parts.append( '[User Input]' )
	preview_parts.append( user_input or '' )
	
	return '\n\n'.join( preview_parts ).strip( )

def build_prompt( user_input: str ) -> str:
	"""
		Purpose:
		--------
		Build a llama.cpp-compatible prompt using unified system instructions, task-specific
		Text Generation settings, optional semantic/basic context, and chat history.

		Parameters:
		-----------
		user_input : str

		Returns:
		--------
		str
	"""
	global embedder
	
	system_instructions = get_effective_system_instructions( )
	task_block = build_task_instruction_block( )
	use_semantic = bool( st.session_state.get( 'use_semantic', False ) )
	use_chat_history = bool( st.session_state.get( 'use_chat_history', True ) )
	use_document_context = bool( st.session_state.get( 'use_document_context', False ) )
	basic_docs = st.session_state.get( 'basic_docs', [ ] )
	messages = st.session_state.get( 'messages', [ ] )
	top_k_value = int( st.session_state.get( 'top_k', 0 ) )
	if top_k_value <= 0:
		top_k_value = 4
	
	system_parts: List[ str ] = [ ]
	if system_instructions:
		system_parts.append( system_instructions )
	if task_block:
		system_parts.append( task_block )
	
	system_text = '\n\n'.join( [ p for p in system_parts if p ] ).strip( )
	prompt = ''
	if system_text:
		prompt += f'<|system|>\n{system_text}\n</s>\n'
	
	if use_semantic:
		if embedder is None:
			embedder = load_embedder( )
		
		if embedder is not None:
			with sqlite3.connect( cfg.DB_PATH ) as conn:
				rows = conn.execute( 'SELECT chunk, vector FROM embeddings' ).fetchall( )
			
			if rows:
				q = embedder.encode( [ user_input ] )[ 0 ]
				scored = [ (c, cosine_similarity( q, np.frombuffer( v ) )) for c, v in rows ]
				for c, _ in sorted( scored, key=lambda x: x[ 1 ], reverse=True )[ :top_k_value ]:
					prompt += f'<|system|>\nSemantic Context:\n{c}\n</s>\n'
	
	if use_document_context and isinstance( basic_docs, list ):
		for d in basic_docs[ :6 ]:
			prompt += f'<|system|>\nDocument Context:\n{d}\n</s>\n'
	
	if use_chat_history and isinstance( messages, list ):
		for msg in messages:
			role = ''
			content = ''
			
			if isinstance( msg, tuple ) or isinstance( msg, list ):
				if len( msg ) == 2:
					role = str( msg[ 0 ] or '' ).strip( )
					content = str( msg[ 1 ] or '' )
			elif isinstance( msg, dict ):
				role = str( msg.get( 'role', '' ) or '' ).strip( )
				content = str( msg.get( 'content', '' ) or '' )
			
			if role in ('user', 'assistant', 'system'):
				prompt += f'<|{role}|>\n{content}\n</s>\n'
	
	prompt += f'<|user|>\n{user_input}\n</s>\n<|assistant|>\n'
	return prompt

def resolve_generation_parameters( ) -> Dict[ str, Any ]:
	"""Generation parameter resolution.

	Purpose:
	    Reads, normalizes, validates, and returns the complete generation-parameter
	    contract captured by the Streamlit user interface.

	Args:
	    None.

	Returns:
	    Dict[str, Any]: Normalized generation parameters shared by local and remote
	    model execution paths.
	"""
	context_window: int = int(
		st.session_state.get( 'context_window', cfg.DEFAULT_CTX, ) or cfg.DEFAULT_CTX )
	
	if context_window <= 0:
		context_window = int( cfg.DEFAULT_CTX )
	
	cpu_threads: int = int( st.session_state.get( 'cpu_threads', cfg.CORES, ) or cfg.CORES )
	
	if cpu_threads <= 0:
		cpu_threads = int( cfg.CORES )
	
	max_tokens: int = int( st.session_state.get( 'max_tokens', 1024, ) or 1024 )
	
	if max_tokens <= 0:
		max_tokens = 1024
	
	temperature: float = float( st.session_state.get( 'temperature', 0.0, ) or 0.0 )
	
	temperature = max( 0.0, temperature, )
	
	top_p: float = float( st.session_state.get( 'top_percent', 0.95, ) or 0.95 )
	
	top_p = min( 1.0, max( 0.0, top_p, ), )
	
	top_k: int = int( st.session_state.get( 'top_k', 40, ) or 40 )
	
	if top_k <= 0:
		top_k = 40
	
	frequency_penalty: float = float( st.session_state.get( 'frequency_penalty', 0.0, ) or 0.0 )
	
	presence_penalty: float = float( st.session_state.get( 'presence_penalty', 0.0, ) or 0.0 )
	
	repeat_penalty: float = float( st.session_state.get( 'repeat_penalty', 1.1, ) or 1.1 )
	
	if repeat_penalty <= 0.0:
		repeat_penalty = 1.1
	
	repeat_window: int = int( st.session_state.get( 'repeat_window', 64, ) or 64 )
	
	if repeat_window < 0:
		repeat_window = 0
	
	random_seed: int = int( st.session_state.get( 'random_seed', -1, ) )
	
	return { 'context_window': context_window, 'cpu_threads': cpu_threads, 'max_tokens':
		max_tokens,
		'temperature': temperature, 'top_p': top_p, 'top_k': top_k,
		'frequency_penalty': frequency_penalty, 'presence_penalty': presence_penalty,
		'repeat_penalty': repeat_penalty, 'repeat_window': repeat_window,
		'random_seed': random_seed, }

def run_model_prompt( prompt: str, temperature: float, top_p: float, repeat_penalty: float,
	max_tokens: int, stream: bool, output: Any | None = None, ) -> str:
	"""Execute a completed prompt through the local language model.

	Purpose:
		Executes a fully constructed prompt through the configured local llama.cpp
		model. The function applies the complete generation-parameter contract,
		maintains shared generation lifecycle state, supports streamed and
		non-streamed output, and preserves partial streamed output when cancellation
		is requested.

	Args:
		prompt: Complete prompt submitted directly to the local model.
		temperature: Sampling-temperature override.
		top_p: Nucleus-sampling probability override.
		repeat_penalty: Repetition-penalty override.
		max_tokens: Maximum-output-token override.
		stream: Indicates whether response text should be streamed.
		output: Optional Streamlit output container used during streaming.

	Returns:
		str: Complete or partially generated response text. An empty string is
		returned when the prompt or local model is unavailable.
	"""
	global llm
	
	prompt_value: str = str( prompt or '' )
	
	if not prompt_value.strip( ):
		return ''
	
	generation_parameters: Dict[ str, Any ] = resolve_generation_parameters( )
	
	context_window_value: int = int( generation_parameters[ 'context_window' ] )
	
	cpu_thread_value: int = int( generation_parameters[ 'cpu_threads' ] )
	
	repeat_window_value: int = int( generation_parameters[ 'repeat_window' ] )
	
	llm = load_llm( ctx=context_window_value, threads=cpu_thread_value,
		repeat_window=repeat_window_value, )
	
	if llm is None:
		st.error( f'Local model unavailable at {cfg.MODEL_PATH}' )
		return ''
	
	max_tokens_value: int = (
		int( max_tokens ) if max_tokens is not None and int( max_tokens ) > 0 else int(
			generation_parameters[ 'max_tokens' ] ))
	
	temperature_value: float = (float( temperature ) if temperature is not None else float(
		generation_parameters[ 'temperature' ] ))
	
	top_p_value: float = (
		float( top_p ) if top_p is not None else float( generation_parameters[ 'top_p' ] ))
	
	repeat_penalty_value: float = (float( repeat_penalty ) if repeat_penalty is not None else
	                               float(
		generation_parameters[ 'repeat_penalty' ] ))
	
	top_k_value: int = int( generation_parameters[ 'top_k' ] )
	
	frequency_penalty_value: float = float( generation_parameters[ 'frequency_penalty' ] )
	
	presence_penalty_value: float = float( generation_parameters[ 'presence_penalty' ] )
	
	random_seed_value: int = int( generation_parameters[ 'random_seed' ] )
	
	seed_value: int | None = (random_seed_value if random_seed_value >= 0 else None)
	
	request_arguments: Dict[ str, Any ] = { 'prompt': prompt_value, 'max_tokens': max_tokens_value,
		'temperature': temperature_value, 'top_p': top_p_value, 'top_k': top_k_value,
		'frequency_penalty': frequency_penalty_value, 'presence_penalty': presence_penalty_value,
		'repeat_penalty': repeat_penalty_value, 'seed': seed_value, 'stop': [ '</s>' ], }
	
	request_id: int = begin_generation( provider='local', )
	
	final_status: str = 'completed'
	
	try:
		if not stream:
			response: Dict[ str, Any ] = llm( stream=False, **request_arguments, )
			
			response_choices: Any = response.get( 'choices', [ ], )
			
			if (not isinstance( response_choices, list ) or len(
				response_choices ) == 0 or not isinstance( response_choices[ 0 ], dict )):
				return ''
			
			return str( response_choices[ 0 ].get( 'text', '', ) or '' ).strip( )
		
		response_buffer: str = ''
		
		if output is None:
			output = st.empty( )
		
		response_stream: Any = llm( stream=True, **request_arguments, )
		
		for response_chunk in response_stream:
			if generation_stop_requested( request_id=request_id, ):
				final_status = 'stopped'
				break
			
			if not isinstance( response_chunk, dict ):
				continue
			
			chunk_choices: Any = response_chunk.get( 'choices', [ ], )
			
			if (not isinstance( chunk_choices, list ) or len(
				chunk_choices ) == 0 or not isinstance( chunk_choices[ 0 ], dict )):
				continue
			
			chunk_text: str = str( chunk_choices[ 0 ].get( 'text', '', ) or '' )
			
			if not chunk_text:
				continue
			
			response_buffer += chunk_text
			
			output.markdown( response_buffer + '▌' )
		
		output.markdown( response_buffer )
		
		return response_buffer.strip( )
	
	except Exception:
		final_status = 'failed'
		raise
	
	finally:
		complete_generation( request_id=request_id, status=final_status, )

def run_llm_turn( user_input: str, temperature: float, top_p: float, repeat_penalty: float,
	max_tokens: int, stream: bool, output: Any | None = None, grounded: bool = False, ) -> str:
	"""Execute one Text Generation model turn.

	Purpose:
		Executes Text Generation through the local model by default. Google-grounded
		Gemini generation is used only when the calling workflow explicitly requests
		grounding and the required Gemini API key is configured. Grounding failures
		fall back visibly to the local model so the application's core functionality
		remains available without an external provider.

	Args:
		user_input: Current Text Generation user request.
		temperature: Sampling-temperature override supplied by the calling workflow.
		top_p: Nucleus-sampling probability override supplied by the calling workflow.
		repeat_penalty: Repetition-penalty override supplied by the calling workflow.
		max_tokens: Maximum-output-token override supplied by the calling workflow.
		stream: Indicates whether response text should be streamed.
		output: Optional Streamlit output container used during streaming.
		grounded: Indicates whether the caller explicitly requests Gemini Google Search
			grounding instead of ordinary local-model execution.

	Returns:
		str: Generated response text. An empty string is returned when the user request
		contains no usable text.
	"""
	user_input_value: str = str( user_input or '' ).strip( )
	if not user_input_value:
		return ''
	
	generation_parameters: Dict[ str, Any ] = resolve_generation_parameters( )
	temperature_value: float = (float( temperature ) if temperature is not None else float(
		generation_parameters[ 'temperature' ] ))
	
	top_p_value: float = (
		float( top_p ) if top_p is not None else float( generation_parameters[ 'top_p' ] ))
	
	repeat_penalty_value: float = (float( repeat_penalty ) if repeat_penalty is not None else
	                               float( generation_parameters[ 'repeat_penalty' ] ))
	
	max_tokens_value: int = (
		int( max_tokens ) if max_tokens is not None and int( max_tokens ) > 0 else int(
			generation_parameters[ 'max_tokens' ] ))
	
	if bool( grounded ):
		if gemini_grounding_available( ):
			try:
				st.session_state[ 'generation_provider' ] = 'gemini'
				st.session_state[ 'generation_status' ] = 'generating'
				st.session_state[ 'gemini_grounding_error' ] = ''
				
				return run_grounded_gemini_turn( user_input=user_input_value, model=str(
					st.session_state.get( 'gemini_grounding_model', 'gemini-2.5-flash-lite', ) ),
					stream=bool( stream ), output=output, )
			except Exception as ex:
				st.session_state[ 'gemini_grounding_error' ] = str( ex )
				st.session_state[ 'generation_provider' ] = 'local'
				st.session_state[ 'generation_status' ] = 'local_fallback'
				st.warning( 'Google grounding was unavailable. '
				            'The request is being completed by the local model.' )
		else:
			st.session_state[ 'gemini_grounding_error' ] = ('GEMINI_API_KEY is not configured.')
			st.session_state[ 'generation_provider' ] = 'local'
			st.session_state[ 'generation_status' ] = 'local_fallback'
			st.warning( 'Google grounding requires GEMINI_API_KEY. '
			            'The request is being completed by the local model.' )
	
	st.session_state[ 'generation_provider' ] = 'local'
	if st.session_state.get( 'generation_status' ) != 'local_fallback':
		st.session_state[ 'generation_status' ] = 'generating'
	
	effective_prompt: str = build_prompt( user_input=user_input_value, )
	
	response_text: str = run_model_prompt( prompt=effective_prompt, temperature=temperature_value,
		top_p=top_p_value, repeat_penalty=repeat_penalty_value, max_tokens=max_tokens_value,
		stream=bool( stream ), output=output, )
	
	if st.session_state.get( 'generation_status' ) != 'local_fallback':
		st.session_state[ 'generation_status' ] = 'completed'
	
	return response_text

def get_prompt_categories( ) -> List[ str ]:
	"""
		Purpose:
		--------
		Return supported prompt categories.

		Parameters:
		-----------
		None

		Returns:
		--------
		List[str]
	"""
	return [
			'General Chat',
			'Reasoning',
			'Coding',
			'Translation',
			'Summarization',
			'Extraction',
			'Document Extraction',
			'OCR',
			'Audio',
			'JSON Output'
	]

def get_prompt_task_types( ) -> List[ str ]:
	"""
		Purpose:
		--------
		Return supported task types.

		Parameters:
		-----------
		None

		Returns:
		--------
		List[str]
	"""
	return [
			'Chat',
			'Reasoning',
			'Coding',
			'Translation',
			'Summarization',
			'Extraction'
	]

def infer_prompt_category( prompt_row: Dict[ str, Any ] | None ) -> str:
	"""
		Purpose:
		--------
		Infer a prompt category from the prompt row content.

		Parameters:
		-----------
		prompt_row : Dict[str, Any] | None

		Returns:
		--------
		str
	"""
	if not isinstance( prompt_row, dict ):
		return 'General Chat'
	
	caption = str( prompt_row.get( 'Caption', '' ) or '' ).lower( )
	name = str( prompt_row.get( 'Name', '' ) or '' ).lower( )
	text = str( prompt_row.get( 'Text', '' ) or '' ).lower( )
	
	blob = f'{caption} {name} {text}'
	
	if 'json' in blob:
		return 'JSON Output'
	if 'ocr' in blob:
		return 'OCR'
	if 'audio' in blob or 'transcrib' in blob:
		return 'Audio'
	if 'document' in blob and 'extract' in blob:
		return 'Document Extraction'
	if 'extract' in blob:
		return 'Extraction'
	if 'summar' in blob:
		return 'Summarization'
	if 'translat' in blob:
		return 'Translation'
	if 'coding' in blob or 'code' in blob or 'debug' in blob or 'refactor' in blob:
		return 'Coding'
	if 'reason' in blob or 'analysis' in blob:
		return 'Reasoning'
	
	return 'General Chat'

def build_starter_prompt_template( category: str, task_type: str, response_format: str,
		language: str ) -> str:
	"""
		Purpose:
		--------
		Build a starter prompt template from high-level prompt metadata.

		Parameters:
		-----------
		category : str
		task_type : str
		response_format : str
		language : str

		Returns:
		--------
		str
	"""
	category_value = str( category or 'General Chat' ).strip( )
	task_value = str( task_type or 'Chat' ).strip( )
	format_value = str( response_format or 'Markdown' ).strip( )
	language_value = str( language or 'English' ).strip( )
	
	lines: List[ str ] = [ ]
	lines.append( f'You are a local AI assistant operating in the category "{category_value}".' )
	lines.append( f'Primary task type: {task_value}.' )
	lines.append( f'Response format: {format_value}.' )
	lines.append( f'Preferred language: {language_value}.' )
	
	if category_value == 'Reasoning':
		lines.append( 'Provide careful, structured analytical answers grounded in the supplied information.' )
	elif category_value == 'Coding':
		lines.append( 'Produce editor-ready code and explain only what is necessary for correct implementation.' )
	elif category_value == 'Translation':
		lines.append( 'Translate faithfully while preserving meaning, tone, and structure.' )
	elif category_value == 'Summarization':
		lines.append( 'Summarize faithfully and preserve key facts, names, and dates.' )
	elif category_value == 'Extraction':
		lines.append( 'Extract only supported facts. Do not invent missing values.' )
	elif category_value == 'Document Extraction':
		lines.append( 'Use the document content as the evidence base and extract structured facts faithfully.' )
	elif category_value == 'OCR':
		lines.append( 'Extract visible text accurately and preserve structural cues where possible.' )
	elif category_value == 'Audio':
		lines.append( 'Work from transcript/audio-derived text and preserve meaning and speaker intent.' )
	elif category_value == 'JSON Output':
		lines.append( 'Return valid JSON only, matching the requested structure exactly.' )
	else:
		lines.append( 'Respond helpfully, accurately, and concisely.' )
	
	lines.append( 'If information is missing, state that clearly.' )
	return '\n'.join( lines ).strip( )

def generate_prompt_template_draft( goal: str, constraints: str, style: str, category: str,
	task_type: str, response_format: str, language: str ) -> str:
	"""
		Purpose:
		--------
		Generate a draft system prompt using the local model.

		Parameters:
		-----------
		goal : str
		constraints : str
		style : str
		category : str
		task_type : str
		response_format : str
		language : str

		Returns:
		--------
		str
	"""
	prompt = f"""
	Create a strong system prompt for a local AI application.
	
	Category: {category}
	Task Type: {task_type}
	Response Format: {response_format}
	Language: {language}
	Goal: {goal}
	Constraints: {constraints}
	Style: {style}
	
	Write only the system prompt text. Do not add explanation.
	""".strip( )
	
	return run_llm_turn( user_input=prompt,
		temperature=float( st.session_state.get( 'temperature', 0.2 ) ),
		top_p=float( st.session_state.get( 'top_percent', 0.95 ) ),
		repeat_penalty=float( st.session_state.get( 'repeat_penalty', 1.05 ) ), max_tokens=512,
		stream=False, output=None )

def apply_prompt_to_text_generation( prompt_text: str ) -> None:
	"""
		Purpose:
		--------
		Apply a prompt to shared Text Generation settings.

		Parameters:
		-----------
		prompt_text : str

		Returns:
		--------
		None
	"""
	st.session_state[ 'system_instructions' ] = str( prompt_text or '' )

def apply_prompt_to_document_qna( prompt_text: str ) -> None:
	"""
		Purpose:
		--------
		Apply a prompt to shared Document Q&A settings.

		Parameters:
		-----------
		prompt_text : str

		Returns:
		--------
		None
	"""
	st.session_state[ 'system_instructions' ] = str( prompt_text or '' )
	st.session_state[ 'require_grounding' ] = True
	st.session_state[ 'answer_from_excerpts_only' ] = True

def apply_prompt_metadata_to_shared_state( category: str, task_type: str,
		response_format: str, language: str ) -> None:
	"""
		Purpose:
		--------
		Apply prompt metadata to the shared app contract.

		Parameters:
		-----------
		category : str
		task_type : str
		response_format : str
		language : str

		Returns:
		--------
		None
	"""
	st.session_state[ 'task_preset' ] = str( task_type or 'Chat' )
	st.session_state[ 'response_format' ] = str( response_format or 'Markdown' )
	st.session_state[ 'translation_target_language' ] = str( language or 'English' )

def clone_prompt_record( source_prompt: Dict[ str, Any ] | None, ) -> None:
	"""Prompt record cloning.

	Purpose:
	    Copies one prompt record into the Prompt Engineering edit surface as a new,
	    unsaved prompt draft.

	Args:
	    source_prompt (Dict[str, Any] | None): Source prompt record containing Caption,
	    Name, Category, and Text values.

	Returns:
	    None: This function performs its work through side effects and does not return
	    a value.
	"""
	if not isinstance( source_prompt, dict ):
		return
	
	source_caption: str = str( source_prompt.get( 'Caption', '', ) or '' ).strip( )
	source_name: str = str( source_prompt.get( 'Name', '', ) or '' ).strip( )
	source_category: str = str( source_prompt.get( 'Category', '', ) or '' ).strip( )
	source_text: str = str( source_prompt.get( 'Text', '', ) or '' )
	st.session_state[ 'pe_selected_id' ] = None
	
	st.session_state[ 'pe_caption' ] = (
		f'{source_caption} Copy' if source_caption else 'Prompt Copy')
	
	st.session_state[ 'pe_name' ] = source_name
	st.session_state[ 'pe_edit_category' ] = source_category
	st.session_state[ 'pe_text' ] = source_text

def reset_prompt_engineering_selection( ) -> None:
	"""Prompt Engineering selection reset.

	Purpose:
	    Clears the selected prompt record and restores the Prompt Engineering edit surface
	    to an empty new-record state.

	Args:
	    None.

	Returns:
	    None: This function performs its work through side effects and does not return
	    a value.
	"""
	st.session_state[ 'pe_selected_id' ] = None
	st.session_state[ 'pe_caption' ] = ''
	st.session_state[ 'pe_name' ] = ''
	st.session_state[ 'pe_edit_category' ] = ''
	st.session_state[ 'pe_text' ] = ''

def load_prompt_into_engineering_state( prompt_id: int, ) -> None:
	"""Prompt Engineering record loading.

	Purpose:
	    Loads one prompt record into the Prompt Engineering edit surface using its
	    integer primary key.

	Args:
	    prompt_id (int): Prompts.ID value of the record to load.

	Returns:
	    None: This function performs its work through side effects and does not return
	    a value.

	Raises:
	    ValueError: Raised when the prompt identifier is invalid or the record does not
	    exist.
	"""
	prompt_id_value: int = int( prompt_id )
	
	if prompt_id_value <= 0:
		raise ValueError( 'A valid prompt ID is required.' )
	
	prompt_record: Dict[ str, Any ] | None = fetch_prompt_by_id( prompt_id=prompt_id_value, )
	
	if prompt_record is None:
		raise ValueError( f'Prompt ID {prompt_id_value} was not found.' )
	
	st.session_state[ 'pe_selected_id' ] = int( prompt_record[ 'ID' ] )
	
	st.session_state[ 'pe_caption' ] = str( prompt_record.get( 'Caption', '', ) or '' ).strip( )
	
	st.session_state[ 'pe_name' ] = str( prompt_record.get( 'Name', '', ) or '' ).strip( )
	
	st.session_state[ 'pe_edit_category' ] = str(
		prompt_record.get( 'Category', '', ) or '' ).strip( )
	
	st.session_state[ 'pe_text' ] = str( prompt_record.get( 'Text', '', ) or '' )

def reset_prompt_engineering_page_on_filter_change( ) -> None:
	"""Prompt Engineering paging reset.

	Purpose:
	    Restores Prompt Engineering pagination to page one when the search text or
	    category filter changes.

	Args:
	    None.

	Returns:
	    None: This function performs its work through side effects and does not return
	    a value.
	"""
	current_search: str = str( st.session_state.get( 'pe_search', '', ) or '' )
	current_category: str = str( st.session_state.get( 'pe_filter_category', '', ) or '' )
	last_search: str = str( st.session_state.get( 'pe_last_search', '', ) or '' )
	last_category: str = str( st.session_state.get( 'pe_last_filter_category', '', ) or '' )
	if (current_search != last_search or current_category != last_category):
		st.session_state[ 'pe_page' ] = 1
	
	st.session_state[ 'pe_last_search' ] = current_search
	st.session_state[ 'pe_last_filter_category' ] = current_category

def reset_prompt_selection( ) -> None:
	"""Prompt selection reset.

	Purpose:
	    Clears the selected database record and restores the prompt edit surface to
	    an empty new-record state.

	Args:
	    None.

	Returns:
	    None: This function performs its work through side effects and does not return
	    a value.
	"""
	st.session_state[ 'pe_selected_id' ] = None
	st.session_state[ 'pe_caption' ] = ''
	st.session_state[ 'pe_name' ] = ''
	st.session_state[ 'pe_edit_category' ] = ''
	st.session_state[ 'pe_text' ] = ''
	st.session_state[ 'pe_generated_template' ] = ''
	st.session_state[ 'pe_table_revision' ] = int(
		st.session_state.get( 'pe_table_revision', 0 ) ) + 1

def load_prompt_record( prompt_id: int ) -> None:
	"""Prompt record loading.

	Purpose:
	    Loads one prompt record into the Prompt Engineering edit surface by its
	    integer primary key.

	Args:
	    prompt_id (int): Prompts.ID value of the record to load.

	Returns:
	    None: This function performs its work through side effects and does not return
	    a value.

	Raises:
	    ValueError: Raised when the identifier is invalid or the record does not exist.
	"""
	prompt_id_value: int = int( prompt_id )
	
	if prompt_id_value <= 0:
		raise ValueError( 'A valid prompt ID is required.' )
	
	prompt_record: Dict[ str, Any ] | None = fetch_prompt_by_id( prompt_id=prompt_id_value, )
	
	if prompt_record is None:
		raise ValueError( f'Prompt ID {prompt_id_value} was not found.' )
	
	st.session_state[ 'pe_selected_id' ] = int( prompt_record[ 'ID' ] )
	
	st.session_state[ 'pe_caption' ] = str( prompt_record.get( 'Caption', '' ) or '' ).strip( )
	
	st.session_state[ 'pe_name' ] = str( prompt_record.get( 'Name', '' ) or '' ).strip( )
	
	st.session_state[ 'pe_edit_category' ] = str(
		prompt_record.get( 'Category', '' ) or '' ).strip( )
	
	st.session_state[ 'pe_text' ] = str( prompt_record.get( 'Text', '' ) or '' )

def apply_prompt_metadata( ) -> None:
	"""Prompt metadata application.

	Purpose:
	    Applies Prompt Engineering task metadata to the shared Text Generation
	    controls without modifying unrelated model or document state.

	Args:
	    None.

	Returns:
	    None: This function performs its work through side effects and does not return
	    a value.
	"""
	st.session_state[ 'task_preset' ] = str(
		st.session_state.get( 'pe_task_type', 'Chat' ) or 'Chat' )
	
	st.session_state[ 'response_format' ] = str(
		st.session_state.get( 'pe_response_format', 'Markdown' ) or 'Markdown' )
	
	st.session_state[ 'translation_target_language' ] = str(
		st.session_state.get( 'pe_language', 'English' ) or 'English' )

def apply_prompt_to_shared_instructions( enable_document_grounding: bool = False, ) -> None:
	"""Shared System Instructions application.

	Purpose:
	    Applies the current Prompt Engineering text and metadata to the shared
	    System Instructions contract.

	Args:
	    enable_document_grounding (bool): Indicates whether Document Q&A grounding
	    controls should also be enabled.

	Returns:
	    None: This function performs its work through side effects and does not return
	    a value.
	"""
	prompt_text: str = str( st.session_state.get( 'pe_text', '' ) or '' )
	
	st.session_state[ 'system_instructions' ] = prompt_text
	apply_prompt_metadata( )
	
	selected_prompt_id: Any = st.session_state.get( 'pe_selected_id' )
	
	st.session_state[ 'selected_prompt_id' ] = selected_prompt_id
	
	st.session_state[ 'active_prompt_caption' ] = str(
		st.session_state.get( 'pe_caption', '' ) or '' ).strip( )
	
	st.session_state[ 'active_prompt_name' ] = str(
		st.session_state.get( 'pe_name', '' ) or '' ).strip( )
	
	if enable_document_grounding:
		st.session_state[ 'require_grounding' ] = True
		st.session_state[ 'answer_from_excerpts_only' ] = True

def clone_current_prompt( ) -> None:
	"""Prompt record cloning.

	Purpose:
	    Copies the currently loaded prompt into the edit surface as a new unsaved
	    prompt record.

	Args:
	    None.

	Returns:
	    None: This function performs its work through side effects and does not return
	    a value.

	Raises:
	    ValueError: Raised when no prompt is loaded into the edit surface.
	"""
	source_caption: str = str( st.session_state.get( 'pe_caption', '' ) or '' ).strip( )
	
	source_name: str = str( st.session_state.get( 'pe_name', '' ) or '' ).strip( )
	
	source_category: str = str( st.session_state.get( 'pe_edit_category', '' ) or '' ).strip( )
	
	source_text: str = str( st.session_state.get( 'pe_text', '' ) or '' )
	
	if not source_caption and not source_name and not source_text.strip( ):
		raise ValueError( 'Load or enter a prompt before creating a clone.' )
	
	st.session_state[ 'pe_selected_id' ] = None
	
	st.session_state[ 'pe_caption' ] = (
		f'{source_caption} Copy' if source_caption else 'Prompt Copy')
	
	st.session_state[ 'pe_name' ] = source_name
	st.session_state[ 'pe_edit_category' ] = source_category
	st.session_state[ 'pe_text' ] = source_text
	st.session_state[ 'pe_generated_template' ] = ''
	
	st.session_state[ 'pe_table_revision' ] = int(
		st.session_state.get( 'pe_table_revision', 0 ) ) + 1
	
# ----------- DATABASE UTILITIES -------------------------

def create_connection( ) -> sqlite3.Connection:
	return sqlite3.connect( cfg.DB_PATH )

def list_tables( ) -> List[ str ]:
	with create_connection( ) as conn:
		_query = "SELECT name FROM sqlite_master WHERE type='table' ORDER BY name;"
		rows = conn.execute( _query ).fetchall( )
		return [ r[ 0 ] for r in rows ]

def create_schema( table: str ) -> List[ Tuple ]:
	with create_connection( ) as conn:
		return conn.execute( f'PRAGMA table_info("{table}");' ).fetchall( )

def read_table( table: str, limit: int | None = None, offset: int = 0, ) -> pd.DataFrame:
	"""SQLite table retrieval.

	Purpose:
	    Retrieves rows from a SQLite table while exposing the internal row identifier
	    under a unique column name that cannot duplicate an INTEGER PRIMARY KEY column.

	Args:
	    table (str): SQLite table name.
	    limit (int | None): Optional maximum number of rows to retrieve.
	    offset (int): Number of rows to skip before retrieval.

	Returns:
	    pd.DataFrame: Retrieved table rows with a uniquely named internal row identifier.

	Raises:
	    ValueError: Raised when the table name is invalid.
	"""
	table_value = str( table or '' ).strip( )
	if not table_value:
		raise ValueError( 'Table name is required.' )
	
	with create_connection( ) as conn:
		schema_rows: List[ Tuple[ Any, ... ] ] = conn.execute(
			f'PRAGMA table_info("{table_value}");' ).fetchall( )
		
		column_names: List[ str ] = [ str( schema_row[ 1 ] ) for schema_row in schema_rows if
			len( schema_row ) > 1 ]
		
		rowid_alias: str = '__rowid__'
		while rowid_alias in column_names:
			rowid_alias = f'_{rowid_alias}'
		
		query: str = (f'SELECT rowid AS "{rowid_alias}", * '
		              f'FROM "{table_value}"')
		
		query_parameters: List[ int ] = [ ]
		if limit is not None:
			limit_value: int = max( 1, int( limit ), )
			offset_value: int = max( 0, int( offset ), )
			query += ' LIMIT ? OFFSET ?'
			query_parameters.extend( [ limit_value, offset_value, ] )
		
		return pd.read_sql_query( query, conn, params=query_parameters, )

def drop_table( table: str ) -> None:
	"""
		Purpose:
		--------
		Safely drop a table if it exists.
	
		Parameters:
		-----------
		table : str
			Table name.
	"""
	if not table:
		return
	
	with create_connection( ) as conn:
		conn.execute( f'DROP TABLE IF EXISTS "{table}";' )
		conn.commit( )

def rename_table( old_name: str, new_name: str ) -> None:
	"""
		Purpose:
		--------
		Rename an existing SQLite table. Attempts native ALTER TABLE rename first; if it fails,
		falls back to a schema-safe rebuild using the original CREATE TABLE statement and
		preserves indexes.

		Parameters:
		-----------
		old_name : str
			Existing table name.

		new_name : str
			New table name.

		Returns:
		--------
		None
	"""
	if not old_name or not new_name:
		return
	
	with create_connection( ) as conn:
		try:
			conn.execute( f'ALTER TABLE "{old_name}" RENAME TO "{new_name}";' )
			conn.commit( )
			return
		except Exception:
			pass
		
		row = conn.execute(
			"""
            SELECT sql
            FROM sqlite_master
            WHERE type ='table' AND name =?
			""",
			(old_name,)
		).fetchone( )
		
		if not row or not row[ 0 ]:
			raise ValueError( "Table definition not found." )
		
		create_sql = row[ 0 ]
		
		indexes = conn.execute(
			"""
            SELECT sql
            FROM sqlite_master
            WHERE type ='index' AND tbl_name=? AND sql IS NOT NULL
			""",
			(old_name,)
		).fetchall( )
		
		open_paren = create_sql.find( "(" )
		if open_paren == -1:
			raise ValueError( "Malformed CREATE TABLE statement." )
		
		temp_name = f"{new_name}__rebuild_temp"
		
		conn.execute( "BEGIN" )
		conn.execute( f'CREATE TABLE "{temp_name}" {create_sql[ open_paren: ]}' )
		
		cols = [ r[ 1 ] for r in conn.execute( f'PRAGMA table_info("{old_name}");' ).fetchall( ) ]
		col_list = ", ".join( [ f'"{c}"' for c in cols ] )
		
		conn.execute(
			f'INSERT INTO "{temp_name}" ({col_list}) SELECT {col_list} FROM "{old_name}";'
		)
		
		conn.execute( f'DROP TABLE "{old_name}";' )
		conn.execute( f'ALTER TABLE "{temp_name}" RENAME TO "{new_name}";' )
		
		for idx in indexes:
			idx_sql = idx[ 0 ]
			if idx_sql:
				idx_sql = idx_sql.replace( f'ON "{old_name}"', f'ON "{new_name}"' )
				conn.execute( idx_sql )
		
		conn.commit( )

def rename_column( table_name: str, old_name: str, new_name: str ) -> None:
	"""
		Purpose:
		--------
		Rename a column within an existing SQLite table. Attempts native ALTER TABLE rename
		first; if it fails, falls back to a schema-safe rebuild preserving column order, data,
		and indexes.

		Parameters:
		-----------
		table_name : str
			Table containing the column.

		old_name : str
			Existing column name.

		new_name : str
			New column name.

		Returns:
		--------
		None
	"""
	if not table_name or not old_name or not new_name:
		return
	
	with create_connection( ) as conn:
		try:
			conn.execute(
				f'ALTER TABLE "{table_name}" RENAME COLUMN "{old_name}" TO "{new_name}";'
			)
			conn.commit( )
			return
		except Exception:
			pass
		
		row = conn.execute(
			"""
            SELECT sql
            FROM sqlite_master
            WHERE type ='table' AND name =?
			""",
			(table_name,)
		).fetchone( )
		
		if not row or not row[ 0 ]:
			raise ValueError( "Table definition not found." )
		
		create_sql = row[ 0 ]
		
		indexes = conn.execute(
			"""
            SELECT sql
            FROM sqlite_master
            WHERE type ='index' AND tbl_name=? AND sql IS NOT NULL
			""",
			(table_name,)
		).fetchall( )
		
		schema = conn.execute( f'PRAGMA table_info("{table_name}");' ).fetchall( )
		cols = [ r[ 1 ] for r in schema ]
		if old_name not in cols:
			raise ValueError( "Column not found." )
		
		mapped_cols = [ (new_name if c == old_name else c) for c in cols ]
		
		temp_table = f"{table_name}__rebuild_temp"
		
		col_defs: List[ str ] = [ ]
		pk_cols = [ r for r in schema if int( r[ 5 ] or 0 ) > 0 ]
		single_pk = len( pk_cols ) == 1
		
		for row in schema:
			col_name = row[ 1 ]
			col_type = row[ 2 ] or ''
			not_null = int( row[ 3 ] or 0 )
			default_value = row[ 4 ]
			pk = int( row[ 5 ] or 0 )
			
			out_name = new_name if col_name == old_name else col_name
			col_def = f'"{out_name}" {col_type}'.strip( )
			
			if not_null:
				col_def += ' NOT NULL'
			
			if default_value is not None:
				col_def += f' DEFAULT {default_value}'
			
			if single_pk and pk == 1:
				col_def += ' PRIMARY KEY'
			
			col_defs.append( col_def )
		
		new_create_sql = f'CREATE TABLE "{temp_table}" ({", ".join( col_defs )});'
		
		old_select = ", ".join( [ f'"{c}"' for c in cols ] )
		new_insert = ", ".join( [ f'"{c}"' for c in mapped_cols ] )
		
		conn.execute( "BEGIN" )
		conn.execute( new_create_sql )
		conn.execute(
			f'INSERT INTO "{temp_table}" ({new_insert}) SELECT {old_select} FROM "{table_name}";'
		)
		
		conn.execute( f'DROP TABLE "{table_name}";' )
		conn.execute( f'ALTER TABLE "{temp_table}" RENAME TO "{table_name}";' )
		
		for idx in indexes:
			idx_sql = idx[ 0 ]
			if idx_sql:
				idx_sql = idx_sql.replace( f'"{old_name}"', f'"{new_name}"' )
				conn.execute( idx_sql )
		
		conn.commit( )

def create_index( table: str, column: str ) -> None:
	"""
		Purpose:
		--------
		Create a safe SQLite index on a specified table column.
	
		Handles:
			- Spaces in column names
			- Special characters
			- Reserved words
			- Duplicate index names
			- Validation against actual table schema
	
		Parameters:
		-----------
		table : str
			Table name.
		column : str
			Column name to index.
	"""
	if not table or not column:
		return
	
	# ----------  Validate table exists
	tables = list_tables( )
	if table not in tables:
		raise ValueError( "Invalid table name." )
	
	# ----------  Validate column exists
	schema = create_schema( table )
	valid_columns = [ col[ 1 ] for col in schema ]
	
	if column not in valid_columns:
		raise ValueError( "Invalid column name." )
	
	# ----------  Sanitize index name (identifier only)
	safe_index_name = re.sub( r"[^0-9a-zA-Z_]+", "_", f"idx_{table}_{column}" )
	
	# ----------  Create index safely (quote identifiers)
	sql = f'CREATE INDEX IF NOT EXISTS "{safe_index_name}" ON "{table}"("{column}");'
	
	with create_connection( ) as conn:
		conn.execute( sql )
		conn.commit( )

def apply_filters( df: pd.DataFrame ) -> pd.DataFrame:
	st.subheader( 'Advanced Filters' )
	col1, col2, col3 = st.columns( 3 )
	column = col1.selectbox( 'Column', df.columns )
	operator = col2.selectbox( 'Operator', [ '=', '!=', '>', '<', '>=', '<=', 'contains' ] )
	value = col3.text_input( 'Value' )
	if value:
		if operator == '=':
			df = df[ df[ column ] == value ]
		elif operator == '!=':
			df = df[ df[ column ] != value ]
		elif operator == '>':
			df = df[ df[ column ].astype( float ) > float( value ) ]
		elif operator == '<':
			df = df[ df[ column ].astype( float ) < float( value ) ]
		elif operator == '>=':
			df = df[ df[ column ].astype( float ) >= float( value ) ]
		elif operator == '<=':
			df = df[ df[ column ].astype( float ) <= float( value ) ]
		elif operator == 'contains':
			df = df[ df[ column ].astype( str ).str.contains( value ) ]
	
	return df

def create_aggregation( df: pd.DataFrame ):
	st.subheader( 'Aggregation Engine' )
	
	numeric_cols = df.select_dtypes( include=[ 'number' ] ).columns.tolist( )
	
	if not numeric_cols:
		st.info( 'No numeric columns available.' )
		return
	
	col = st.selectbox( 'Column', numeric_cols )
	agg = st.selectbox( 'Aggregation', [ 'COUNT', 'SUM', 'AVG', 'MIN', 'MAX', 'MEDIAN' ] )
	
	if agg == 'COUNT':
		result = df[ col ].count( )
	elif agg == 'SUM':
		result = df[ col ].sum( )
	elif agg == 'AVG':
		result = df[ col ].mean( )
	elif agg == 'MIN':
		result = df[ col ].min( )
	elif agg == 'MAX':
		result = df[ col ].max( )
	elif agg == 'MEDIAN':
		result = df[ col ].median( )
	
	st.metric( 'Result', result )

def create_visualization( df: pd.DataFrame ):
	st.subheader( 'Visualization Engine' )
	numeric_cols = df.select_dtypes( include=[ 'number' ] ).columns.tolist( )
	categorical_cols = df.select_dtypes( include=[ 'object' ] ).columns.tolist( )
	chart = st.selectbox( 'Chart Type',
		[ 'Histogram', 'Bar', 'Line', 'Scatter', 'Box', 'Pie', 'Correlation' ] )
	
	if chart == 'Histogram' and numeric_cols:
		col = st.selectbox( 'Column', numeric_cols )
		fig = px.histogram( df, x=col )
		st.plotly_chart( fig, use_container_width=True )
	
	elif chart == 'Bar':
		x = st.selectbox( 'X', df.columns )
		y = st.selectbox( 'Y', numeric_cols )
		fig = px.bar( df, x=x, y=y )
		st.plotly_chart( fig, use_container_width=True )
	
	elif chart == 'Line':
		x = st.selectbox( 'X', df.columns )
		y = st.selectbox( 'Y', numeric_cols )
		fig = px.line( df, x=x, y=y )
		st.plotly_chart( fig, use_container_width=True )
	
	elif chart == 'Scatter':
		x = st.selectbox( 'X', numeric_cols )
		y = st.selectbox( 'Y', numeric_cols )
		fig = px.scatter( df, x=x, y=y )
		st.plotly_chart( fig, use_container_width=True )
	
	elif chart == 'Box':
		col = st.selectbox( 'Column', numeric_cols )
		fig = px.box( df, y=col )
		st.plotly_chart( fig, use_container_width=True )
	
	elif chart == 'Pie':
		col = st.selectbox( 'Category Column', categorical_cols )
		fig = px.pie( df, names=col )
		st.plotly_chart( fig, use_container_width=True )
	
	elif chart == 'Correlation' and len( numeric_cols ) > 1:
		corr = df[ numeric_cols ].corr( )
		fig = px.imshow( corr, text_auto=True )
		st.plotly_chart( fig, use_container_width=True )

def dm_create_table_from_df( table_name: str, df: pd.DataFrame ):
	columns = [ ]
	for col in df.columns:
		sql_type = get_sqlite_type( df[ col ].dtype )
		safe_col = col.replace( ' ', '_' )
		columns.append( f'{safe_col} {sql_type}' )
	
	create_stmt = f'CREATE TABLE IF NOT EXISTS {table_name} ({", ".join( columns )});'
	
	with create_connection( ) as conn:
		conn.execute( create_stmt )
		conn.commit( )

def insert_data( table_name: str, df: pd.DataFrame ):
	df = df.copy( )
	df.columns = [ c.replace( ' ', '_' ) for c in df.columns ]
	
	placeholders = ', '.join( [ '?' ] * len( df.columns ) )
	stmt = f'INSERT INTO {table_name} VALUES ({placeholders});'
	
	with create_connection( ) as conn:
		conn.executemany( stmt, df.values.tolist( ) )
		conn.commit( )

def get_sqlite_type( dtype ) -> str:
	"""
		Purpose:
		--------
		Map a pandas dtype to an appropriate SQLite column type.
	
		Parameters:
		-----------
		dtype : pandas dtype
			The dtype of a pandas Series.
	
		Returns:
		--------
		str
			SQLite column type.
	"""
	dtype_str = str( dtype ).lower( )
	
	# ----------  Integer Types
	if "int" in dtype_str:
		return "INTEGER"
	
	# ----------  Float Types
	if "float" in dtype_str:
		return "REAL"
	
	# ----------  Boolean
	if "jimil" in dtype_str:
		return "INTEGER"
	
	# ----------  Datetime
	if "datetime" in dtype_str:
		return "TEXT"
	
	# ----------  Categorical
	if "category" in dtype_str:
		return "TEXT"
	
	# ----------  Default fallback
	return "TEXT"

def create_custom_table( table_name: str, columns: list ) -> None:
	"""
		Purpose:
		--------
		Create a custom SQLite table from column definitions.
	
		Parameters:
		-----------
		table_name : str
			Name of table.
	
		columns : list of dict
			[
				{
					"name": str,
					"type": str,
					"not_null": jimil,
					"primary_key": jimil,
					"auto_increment": jimil
				}
			]
	"""
	if not table_name:
		raise ValueError( "Table name required." )
	
	# ----------  Validate identifier
	if not re.match( r"^[A-Za-z_][A-Za-z0-9_]*$", table_name ):
		raise ValueError( "Invalid table name." )
	
	col_defs = [ ]
	for col in columns:
		col_name = col[ "name" ]
		col_type = col[ "type" ].upper( )
		if not re.match( r"^[A-Za-z_][A-Za-z0-9_]*$", col_name ):
			raise ValueError( f"Invalid column name: {col_name}" )
		
		definition = f'"{col_name}" {col_type}'
		if col[ "primary_key" ]:
			definition += " PRIMARY KEY"
			if col[ "auto_increment" ] and col_type == "INTEGER":
				definition += " AUTOINCREMENT"
		
		if col[ "not_null" ]:
			definition += " NOT NULL"
		
		col_defs.append( definition )
	
	sql = f'CREATE TABLE IF NOT EXISTS "{table_name}" ({", ".join( col_defs )});'
	with create_connection( ) as conn:
		conn.execute( sql )
		conn.commit( )

def is_safe_query( query: str ) -> bool:
	"""
	
		Purpose:
		--------
		Determine whether a SQL query is read-only and safe to execute.
	
		Allows:
			SELECT
			WITH (CTE returning SELECT)
			EXPLAIN SELECT
			PRAGMA (read-only)
	
		Blocks:
			INSERT, UPDATE, DELETE, DROP, ALTER, CREATE, ATTACH,
			DETACH, VACUUM, REPLACE, TRIGGER, and multiple statements.
			
	"""
	if not query or not isinstance( query, str ):
		return False
	
	q = query.strip( ).lower( )
	
	# ----------  Block multiple statements
	if ';' in q[ :-1 ]:
		return False
	
	# ----------  Remove SQL comments
	q = re.sub( r"--.*?$", "", q, flags=re.MULTILINE )
	q = re.sub( r"/\*.*?\*/", "", q, flags=re.DOTALL )
	q = q.strip( )
	
	# ----------  Allowed starting keywords
	allowed_starts = ('select', 'with', 'explain', 'pragma')
	if not q.startswith( allowed_starts ):
		return False
	
	# ----------  Block dangerous keywords anywhere
	blocked_keywords = ('insert ', 'update ', 'delete ', 'drop ', 'alter ',
	                    'create ', 'attach ', 'detach ', 'vacuum ', 'replace ', 'trigger ')
	
	for keyword in blocked_keywords:
		if keyword in q:
			return False
	
	return True

def create_identifier( name: str ) -> str:
	"""
	
		Purpose:
		--------
		Sanitize a string into a safe SQLite identifier.
	
		- Replaces invalid characters with underscores
		- Ensures it starts with a letter or underscore
		- Prevents empty names
		
	"""
	if not name or not isinstance( name, str ):
		raise ValueError( 'Invalid Identifier.' )
	
	safe = re.sub( r'[^0-9a-zA-Z_]', '_', name.strip( ) )
	if not re.match( r'^[A-Za-z_]', safe ):
		safe = f'_{safe}'
	
	if not safe:
		raise ValueError( 'Invalid identifier after sanitization.' )
	
	return safe

def get_indexes( table: str ):
	with create_connection( ) as conn:
		rows = conn.execute( f'PRAGMA index_list("{table}");' ).fetchall( )
		return rows

def add_column( table: str, column: str, col_type: str ):
	column = create_identifier( column )
	col_type = col_type.upper( )
	
	with create_connection( ) as conn:
		conn.execute(
			f'ALTER TABLE "{table}" ADD COLUMN "{column}" {col_type};' )
		conn.commit( )

def create_profile_table( table: str ):
	df = read_table( table )
	profile_rows = [ ]
	total_rows = len( df )
	for col in df.columns:
		series = df[ col ]
		null_count = series.isna( ).sum( )
		distinct_count = series.nunique( dropna=True )
		row = \
			{
					'column': col, 'dtype': str( series.dtype ),
					'null_%': round( (null_count / total_rows) * 100, 2 ) if total_rows else 0,
					'distinct_%': round( (
							                     distinct_count / total_rows) * 100,
						2 ) if total_rows else 0,
			}
		
		if pd.api.types.is_numeric_dtype( series ):
			row[ "min" ] = series.min( )
			row[ "max" ] = series.max( )
			row[ "mean" ] = series.mean( )
		else:
			row[ "min" ] = None
			row[ "max" ] = None
			row[ "mean" ] = None
		
		profile_rows.append( row )
	
	return pd.DataFrame( profile_rows )

def drop_column( table: str, column: str ):
	if not table or not column:
		raise ValueError( "Table and column required." )
	
	with create_connection( ) as conn:
		schema = conn.execute( f'PRAGMA table_info("{table}");' ).fetchall( )
		if not schema:
			raise ValueError( "Table definition not found." )
		
		col_names = [ r[ 1 ] for r in schema ]
		if column not in col_names:
			raise ValueError( "Column not found." )
		
		remaining = [ r for r in schema if r[ 1 ] != column ]
		if not remaining:
			raise ValueError( "Cannot drop the only remaining column." )
		
		temp_table = f"{table}_rebuild_temp"
		
		pk_cols = [ r for r in remaining if int( r[ 5 ] or 0 ) > 0 ]
		single_pk = len( pk_cols ) == 1
		
		new_defs: List[ str ] = [ ]
		for row in remaining:
			col_name = row[ 1 ]
			col_type = row[ 2 ] or ''
			not_null = int( row[ 3 ] or 0 )
			default_value = row[ 4 ]
			pk = int( row[ 5 ] or 0 )
			
			col_def = f'"{col_name}" {col_type}'.strip( )
			
			if not_null:
				col_def += ' NOT NULL'
			
			if default_value is not None:
				col_def += f' DEFAULT {default_value}'
			
			if single_pk and pk == 1:
				col_def += ' PRIMARY KEY'
			
			new_defs.append( col_def )
		
		new_create_sql = f'CREATE TABLE "{temp_table}" ({", ".join( new_defs )});'
		
		conn.execute( "BEGIN" )
		conn.execute( new_create_sql )
		
		remaining_cols = [ r[ 1 ] for r in remaining ]
		col_list = ", ".join( [ f'"{c}"' for c in remaining_cols ] )
		
		conn.execute(
			f'INSERT INTO "{temp_table}" ({col_list}) '
			f'SELECT {col_list} FROM "{table}";'
		)
		
		indexes = conn.execute(
			"""
            SELECT sql
            FROM sqlite_master
            WHERE type ='index' AND tbl_name=? AND sql IS NOT NULL
			""",
			(table,)
		).fetchall( )
		
		conn.execute( f'DROP TABLE "{table}";' )
		conn.execute( f'ALTER TABLE "{temp_table}" RENAME TO "{table}";' )
		
		for idx in indexes:
			idx_sql = idx[ 0 ]
			if idx_sql and column not in idx_sql:
				conn.execute( idx_sql )
		
		conn.commit( )

# ------------- DOCQNA UTILITIES ----------------------

def get_document_task_instruction( task_name: str, ) -> str:
	"""Document task instruction retrieval.

	Purpose:
	    Returns the task-specific instruction associated with the selected Document Q&A
	    radio-button option.

	Args:
	    task_name (str): Selected Document Q&A task.

	Returns:
	    str: Task-specific instruction used by the document-grounding prompt.

	Raises:
	    ValueError: Raised when the selected task is unsupported.
	"""
	task_value: str = str( task_name or 'Ask a Question' ).strip( )
	
	task_instruction_map: Dict[ str, str ] = { 'Ask a Question': (
		'Answer the user question directly using the retrieved document excerpts.'),
		'Summarize': ('Provide a clear, structured summary of the active document or documents.'),
		'Key Points': ('Extract the principal findings, requirements, decisions, '
		               'responsibilities, '
		               'deadlines, risks, and supporting facts.'),
		'Outline': ('Generate a structured outline that preserves the source hierarchy, major '
		            'sections, topics, and supporting subtopics.'),
		'Entities': ('Extract material people, organizations, programs, laws, regulations, dates, '
		             'locations, systems, financial amounts, and other significant entities.'),
		'Tables': ('Identify and extract material tables and tabular information while preserving '
		           'labels, units, row relationships, column meanings, totals, and explanatory '
		           'context.'),
		'Compare': ('Compare the active documents and identify material similarities, '
		            'differences, '
		            'conflicts, changes, omissions, and relationships.'), }
	
	if task_value not in task_instruction_map:
		raise ValueError( f'Unsupported document task: {task_value}.' )
	
	return task_instruction_map[ task_value ]

def build_document_instruction_block( ) -> str:
	"""Document instruction block construction.

	Purpose:
	    Builds the shared Document Q&A instruction block from the selected document
	    task, grounding controls, and response-format setting.

	Args:
	    None.

	Returns:
	    str: Complete Document Q&A instruction block.

	Raises:
	    ValueError: Raised when the selected Document Q&A task is unsupported.
	"""
	require_grounding: bool = bool( st.session_state.get( 'require_grounding', True, ) )
	answer_from_excerpts_only: bool = bool( st.session_state.get( 'answer_from_excerpts_only', True, ) )
	response_format: str = str( st.session_state.get( 'response_format', 'Markdown', ) or 'Markdown' ).strip( )
	document_task = str( st.session_state.get( 'docqna_task', 'Ask a Question', ) or 'Ask a Question' ).strip( )
	task_instruction = get_document_task_instruction( task_name=document_task, )
	lines = [ 'Document Q&A Instructions:', f'- Task: {document_task}',
		f'- Response Format: {response_format}', f'- Task Guidance: {task_instruction}', ]
	
	if require_grounding:
		lines.append( '- Ground every material statement in the retrieved document excerpts.' )
	
	if answer_from_excerpts_only:
		lines.append( '- When the retrieved excerpts do not support the requested answer, state '
		              'clearly that the available document evidence is insufficient.' )
	
	if response_format == 'JSON':
		lines.append( '- Return valid JSON only.' )
	
	return '\n'.join( lines ).strip( )

def extract_text_from_bytes( file_bytes: bytes, file_name: str = '' ) -> str:
	"""
		Purpose:
		--------
		Extract text from PDF or text-based documents using the current document parsing settings.

		Parameters:
		-----------
		file_bytes : bytes
		file_name : str

		Returns:
		--------
		str
	"""
	if not file_bytes:
		return ''
	
	file_name_value = str( file_name or '' ).lower( )
	include_page_markers = bool( st.session_state.get( 'include_page_markers', False ) )
	prefer_native_pdf_text = bool( st.session_state.get( 'prefer_native_pdf_text', True ) )
	
	try:
		if file_name_value.endswith( '.pdf' ) or file_name_value == '':
			if prefer_native_pdf_text:
				import fitz
				
				doc = fitz.open( stream=file_bytes, filetype='pdf' )
				parts: List[ str ] = [ ]
				page_index = 0
				for page in doc:
					page_index += 1
					page_text = page.get_text( 'text' ) or ''
					if include_page_markers:
						parts.append( f'[Page {page_index}]' )
					parts.append( page_text )
				return '\n'.join( parts ).strip( )
	except Exception:
		pass
	
	try:
		return file_bytes.decode( errors='ignore' ).strip( )
	except Exception:
		return ''

def route_document_query( prompt: str, ) -> str:
	"""Document query routing.

	Purpose:
	    Builds a document-grounded prompt from the supplied request and routes the
	    completed prompt directly through the raw local-model execution layer.

	Args:
	    prompt (str): Document question or predefined document-task instruction.

	Returns:
	    str: Generated document-grounded response.
	"""
	prompt_value: str = str( prompt or '' ).strip( )
	
	if not prompt_value:
		return ''
	
	document_prompt: str = build_document_user_input( user_query=prompt_value,
		k=int( st.session_state.get( 'retrieval_k', 6, ) ), )
	
	if not document_prompt:
		return ''
	
	return run_model_prompt( prompt=document_prompt,
		temperature=float( st.session_state.get( 'temperature', 0.0, ) ),
		top_p=float( st.session_state.get( 'top_percent', 0.95, ) ),
		repeat_penalty=float( st.session_state.get( 'repeat_penalty', 1.1, ) ),
		max_tokens=(int( st.session_state.get( 'max_tokens', 1024, ) ) or 1024), stream=False,
		output=None, )

def summarize_active_document( ) -> str:
	"""
		Purpose:
		--------
		Summarize the currently active document set using the document routing layer.

		Parameters:
		-----------
		None

		Returns:
		--------
		str
	"""
	system_instructions = get_effective_system_instructions( )
	summary_prompt = """
		Provide a clear, structured summary of the active document set.
		Include:
		- Purpose
		- Key themes
		- Major conclusions
		- Important data points
		- Open questions or uncertainties
	"""
	
	if system_instructions:
		summary_prompt = f'{system_instructions}\n\n{summary_prompt}'
	
	return route_document_query( summary_prompt.strip( ) )

def compute_fingerprint( active_docs: List[ str ], doc_bytes: Dict[ str, bytes ] ) -> str:
	'''
		
		Purpose:
		--------
		Computes a stable fingerprint for the currently selected active documents and their byte contents.
	
		Parameters:
		-----------
		active_docs:
			A List[ str ] of active document names.
		doc_bytes:
			A Dict[ str, bytes ] mapping document name to file bytes.
	
		Returns:
		--------
		A str fingerprint suitable for cache invalidation.
	
	'''
	h = hashlib.sha256( )
	for name in sorted( active_docs ):
		b = doc_bytes.get( name, b'' )
		h.update( name.encode( 'utf-8', errors='ignore' ) )
		h.update( len( b ).to_bytes( 8, 'little', signed=False ) )
		h.update( hashlib.sha256( b ).digest( ) )
	return h.hexdigest( )

def extract_text( file_bytes: bytes, file_name: str = '' ) -> str:
	"""
		Purpose:
		--------
		Extract document text using the configured parsing behavior.

		Parameters:
		-----------
		file_bytes : bytes
		file_name : str

		Returns:
		--------
		str
	"""
	return extract_text_from_bytes( file_bytes=file_bytes, file_name=file_name )

def load_sqlite_vec( conn: sqlite3.Connection ) -> bool:
	'''
		
		Purpose:
		--------
		Attempts to load sqlite-vec into the provided SQLite connection.
	
		Parameters:
		-----------
		conn:
			The sqlite3.Connection.
	
		Returns:
		--------
		True if sqlite-vec loaded successfully; otherwise False.
		
	'''
	try:
		import sqlite_vec
		
		sqlite_vec.load( conn )
		return True
	except Exception:
		return False

def ensure_schema( dim: int ) -> bool:
	'''
	
		Purpose:
		--------
		Creates the sqlite-vec virtual table used for Document Q&A embeddings if possible.
	
		Parameters:
		-----------
		dim:
			The embedding dimension (e.g., 384 for all-MiniLM-L6-v2).
	
		Returns:
		--------
		True if the schema exists and is usable; otherwise False.
	
	'''
	conn = create_connection( )
	try:
		ok = load_sqlite_vec( conn )
		if not ok:
			return False
		
		cur = conn.cursor( )
		cur.execute(
			f'''
			CREATE VIRTUAL TABLE IF NOT EXISTS docqna_vec
			USING vec0(
				embedding float[{int( dim )}],
				doc_name TEXT,
				chunk TEXT
			);
			'''
		)
		conn.commit( )
		return True
	except Exception:
		return False
	finally:
		conn.close( )

def build_document_inventory_rows( ) -> List[ Dict[ str, Any ] ]:
	"""
		Purpose:
		--------
		Build inventory rows for the currently active uploaded documents.

		Parameters:
		-----------
		None

		Returns:
		--------
		List[Dict[str, Any]]
	"""
	rows: List[ Dict[ str, Any ] ] = [ ]
	active_docs = st.session_state.get( 'active_docs', [ ] )
	doc_bytes = st.session_state.get( 'doc_bytes', { } )
	
	for name in active_docs:
		b = doc_bytes.get( name, b'' )
		text = extract_text( b, name ) if b else ''
		chunks = chunk_text( text ) if text else [ ]
		
		rows.append(
			{
					'Name': name,
					'SizeBytes': len( b ) if b else 0,
					'TextLength': len( text ) if text else 0,
					'ChunkCount': len( chunks ),
					'Loaded': bool( b ),
			}
		)
	
	return rows

def get_active_document_names_text( ) -> str:
	"""
		Purpose:
		--------
		Build a human-readable string of active document names.

		Parameters:
		-----------
		None

		Returns:
		--------
		str
	"""
	active_docs = st.session_state.get( 'active_docs', [ ] )
	if not isinstance( active_docs, list ) or len( active_docs ) == 0:
		return 'No active documents'
	return ', '.join( [ str( name ) for name in active_docs ] )

def rebuild_index( embedder: Any | None ) -> None:
	"""
		Purpose:
		--------
		Build or refresh the Document Q&A vector index when active documents or chunk settings change.

		Parameters:
		-----------
		embedder : Any | None

		Returns:
		--------
		None
	"""
	if embedder is None:
		st.session_state[ 'docqna_vec_ready' ] = False
		st.session_state[ 'docqna_chunk_count' ] = 0
		st.session_state[ 'docqna_fallback_rows' ] = [ ]
		st.session_state[ 'doc_inventory_rows' ] = build_document_inventory_rows( )
		return
	
	active_docs: List[ str ] = st.session_state.get( 'active_docs', [ ] )
	doc_bytes: Dict[ str, bytes ] = st.session_state.get( 'doc_bytes', { } )
	retrieval_chunk_size = int( st.session_state.get( 'retrieval_chunk_size', 1200 ) )
	retrieval_chunk_overlap = int( st.session_state.get( 'retrieval_chunk_overlap', 200 ) )
	
	fp_seed = f'{retrieval_chunk_size}|{retrieval_chunk_overlap}|'
	fp_seed += compute_fingerprint( active_docs, doc_bytes )
	fp = hashlib.sha256( fp_seed.encode( 'utf-8', errors='ignore' ) ).hexdigest( )
	
	if fp and fp == st.session_state.get( 'docqna_fingerprint', '' ):
		st.session_state[ 'doc_inventory_rows' ] = build_document_inventory_rows( )
		return
	
	st.session_state[ 'docqna_fingerprint' ] = fp
	st.session_state[ 'docqna_chunk_count' ] = 0
	st.session_state[ 'docqna_fallback_rows' ] = [ ]
	st.session_state[ 'doc_inventory_rows' ] = build_document_inventory_rows( )
	
	dim_value = getattr( embedder, 'get_sentence_embedding_dimension', lambda: 384 )( )
	dim = int( dim_value ) if dim_value else 384
	
	prefer_sqlite_vec = bool( st.session_state.get( 'prefer_sqlite_vec', True ) )
	vec_ready = False
	if prefer_sqlite_vec:
		vec_ready = ensure_schema( dim )
	
	st.session_state[ 'docqna_vec_ready' ] = bool( vec_ready )
	
	conn = create_connection( )
	try:
		cur = conn.cursor( )
		
		if vec_ready:
			try:
				cur.execute( 'DELETE FROM docqna_vec;' )
				conn.commit( )
			except Exception:
				st.session_state[ 'docqna_vec_ready' ] = False
				vec_ready = False
		
		total_chunks = 0
		fallback_rows: List[ Tuple[ str, str, bytes ] ] = [ ]
		
		for name in active_docs:
			b = doc_bytes.get( name )
			if not b:
				continue
			
			text = extract_text( b, name )
			if not text:
				continue
			
			chunks = chunk_text(
				text,
				size=retrieval_chunk_size,
				overlap=retrieval_chunk_overlap
			)
			if not chunks:
				continue
			
			vecs = embedder.encode( chunks, show_progress_bar=False )
			vecs = np.asarray( vecs, dtype=np.float32 )
			
			if vec_ready:
				for chunk_text_value, v in zip( chunks, vecs ):
					cur.execute(
						'INSERT INTO docqna_vec ( embedding, doc_name, chunk ) VALUES ( ?, ?, ? );',
						(v.tobytes( ), name, chunk_text_value)
					)
			else:
				for chunk_text_value, v in zip( chunks, vecs ):
					fallback_rows.append( (name, chunk_text_value, v.tobytes( )) )
			
			total_chunks += int( len( chunks ) )
		
		conn.commit( )
		st.session_state[ 'docqna_chunk_count' ] = total_chunks
		
		if not vec_ready:
			st.session_state[ 'docqna_fallback_rows' ] = fallback_rows
		else:
			st.session_state[ 'docqna_fallback_rows' ] = [ ]
	except Exception:
		st.session_state[ 'docqna_vec_ready' ] = False
		st.session_state[ 'docqna_chunk_count' ] = 0
		st.session_state[ 'docqna_fallback_rows' ] = [ ]
	finally:
		conn.close( )

def retrieve_chunks( query: str, k: int | None = None, ) -> List[ Tuple[ str, str, float ] ]:
	"""Document chunk retrieval.

	Purpose:
	    Retrieves the most relevant document chunks using SQLite vector search when
	    available and cosine-similarity fallback when permitted.

	Args:
	    query (str): Search query used to retrieve relevant document chunks.
	    k (int | None): Optional maximum number of chunks to return.

	Returns:
	    List[Tuple[str, str, float]]: Document name, chunk text, and relevance value
	    for each retrieved result.
	"""
	query_value: str = str( query or '' ).strip( )
	
	if not query_value:
		st.session_state[ 'doc_last_retrieval_hits' ] = [ ]
		return [ ]
	
	embedder: Any | None = load_embedder( )
	
	rebuild_index( embedder=embedder, )
	
	if embedder is None:
		st.session_state[ 'doc_last_retrieval_hits' ] = [ ]
		return [ ]
	
	k_value: int = (int( k ) if k is not None else int( st.session_state.get( 'retrieval_k', 6,
	) ))
	
	if k_value <= 0:
		k_value = 6
	
	query_vectors: np.ndarray = np.asarray(
		embedder.encode( [ query_value ], show_progress_bar=False, ), dtype=np.float32, )
	
	if (query_vectors.ndim != 2 or query_vectors.shape[ 0 ] == 0 or query_vectors.shape[ 1 ] == 0):
		st.session_state[ 'doc_last_retrieval_hits' ] = [ ]
		return [ ]
	
	query_vector: np.ndarray = query_vectors[ 0 ]
	
	if bool( st.session_state.get( 'docqna_vec_ready', False, ) ):
		with create_connection( ) as conn:
			try:
				if not load_sqlite_vec( conn=conn, ):
					raise RuntimeError( 'The SQLite vector extension could not be loaded.' )
				
				cursor: sqlite3.Cursor = conn.cursor( )
				
				cursor.execute( """
                                SELECT doc_name,
                                       chunk,
                                       distance
                                FROM docqna_vec
                                WHERE embedding MATCH ?
                                ORDER BY distance ASC LIMIT ?;
				                """, (query_vector.tobytes( ), k_value,), )
				
				vector_rows: List[ Tuple[ Any, ... ] ] = (cursor.fetchall( ))
				
				vector_results: List[ Tuple[ str, str, float ] ] = [
					(str( row[ 0 ] or '' ), str( row[ 1 ] or '' ), float( row[ 2 ] ),) for row in
					vector_rows if len( row ) >= 3 ]
				
				st.session_state[ 'doc_last_retrieval_hits' ] = (vector_results)
				
				return vector_results
			
			except Exception:
				st.session_state[ 'docqna_vec_ready' ] = False
	
	if not bool( st.session_state.get( 'allow_similarity_fallback', True, ) ):
		st.session_state[ 'doc_last_retrieval_hits' ] = [ ]
		return [ ]
	
	fallback_rows: Any = st.session_state.get( 'docqna_fallback_rows', [ ], )
	
	if not isinstance( fallback_rows, list, ):
		st.session_state[ 'doc_last_retrieval_hits' ] = [ ]
		return [ ]
	
	similarity_results: List[ Tuple[ str, str, float ] ] = [ ]
	
	for fallback_row in fallback_rows:
		if (not isinstance( fallback_row, (tuple, list), ) or len( fallback_row ) < 3):
			continue
		
		document_name: str = str( fallback_row[ 0 ] or '' ).strip( )
		
		chunk_value: str = str( fallback_row[ 1 ] or '' )
		
		vector_blob: Any = fallback_row[ 2 ]
		
		if (not document_name or not chunk_value.strip( ) or not isinstance( vector_blob,
			(bytes, bytearray, memoryview), )):
			continue
		
		stored_vector: np.ndarray = np.frombuffer( vector_blob, dtype=np.float32, )
		
		if (stored_vector.size == 0 or stored_vector.shape != query_vector.shape):
			continue
		
		similarity_score: float = cosine_similarity( query_vector, stored_vector, )
		
		similarity_results.append( (document_name, chunk_value, similarity_score,) )
	
	similarity_results.sort( key=lambda result: result[ 2 ], reverse=True, )
	
	similarity_results = similarity_results[ :k_value ]
	
	st.session_state[ 'doc_last_retrieval_hits' ] = (similarity_results)
	
	return similarity_results

def build_document_user_input( user_query: str, k: int | None = None ) -> str:
	"""
		Purpose:
		--------
		Build a document-grounded prompt using retrieved excerpts and the current document action.

		Parameters:
		-----------
		user_query : str
		k : int | None

		Returns:
		--------
		str
	"""
	system = get_effective_system_instructions( )
	doc_instruction_block = build_document_instruction_block( )
	hits = retrieve_chunks( user_query, k=k )
	st.session_state[ 'doc_last_retrieval_hits' ] = hits
	
	context_blocks: List[ str ] = [ ]
	for doc_name, chunk, score in hits:
		context_blocks.append( f'[Document: {doc_name}]\n{chunk}'.strip( ) )
	
	context = '\n\n'.join( context_blocks ).strip( )
	active_doc_names = get_active_document_names_text( )
	
	prompt_parts: List[ str ] = [ ]
	
	if system:
		prompt_parts.append( system )
	
	if doc_instruction_block:
		prompt_parts.append( doc_instruction_block )
	
	prompt_parts.append( f'Active Documents:\n{active_doc_names}' )
	
	if context:
		prompt_parts.append(
			'Use the following retrieved document excerpts as the evidence base for your answer.\n\n'
			f'{context}'
		)
	else:
		prompt_parts.append(
			'No retrieved document excerpts were available for this question.'
		)
	
	prompt_parts.append( f'User Request:\n{user_query}\n\nAnswer:' )
	return '\n\n'.join( prompt_parts ).strip( )

def build_document_task_prompt( task: str, refinement: str, ) -> str:
	"""Document task prompt construction.

	Purpose:
	    Converts the selected Document Q&A task and optional user refinement into
	    a complete instruction for the existing document-query execution pipeline.

	Args:
	    task (str): Selected Document Q&A task.
	    refinement (str): Optional user-supplied task scope or criteria.

	Returns:
	    str: Complete prompt submitted through the Document Q&A execution pipeline.

	Raises:
	    ValueError: Raised when the selected task is unsupported or an Ask a Question
	    submission does not contain a question.
	"""
	task_value: str = str( task or 'Ask a Question' ).strip( )
	
	refinement_value: str = str( refinement or '' ).strip( )
	
	if task_value == 'Ask a Question':
		if not refinement_value:
			raise ValueError( 'Enter a document question before submitting.' )
		
		return refinement_value
	
	task_prompts: Dict[ str, str ] = {
		'Summarize': ('Summarize the active document or documents. Preserve the principal '
		              'purpose, '
		              'structure, material facts, requirements, decisions, conclusions, dates, '
		              'authorities, and quantitative information. Do not introduce information '
		              'that '
		              'is not supported by the retrieved document excerpts.'),
		'Key Points': ('Extract the key points from the active document or documents. Identify '
		               'the '
		               'principal findings, requirements, decisions, responsibilities, deadlines, '
		               'risks, limitations, and supporting facts. Organize the response by topic '
		               'and '
		               'remain grounded in the retrieved document excerpts.'),
		'Outline': ('Generate a structured outline of the active document or documents. Preserve '
		            'the source hierarchy, major sections, important topics, and supporting '
		            'subtopics. Do not invent sections or relationships that are not supported by '
		            'the retrieved document excerpts.'),
		'Entities': ('Extract the material named entities from the active document or documents. '
		             'Identify people, organizations, programs, laws, regulations, dates, '
		             'locations, '
		             'systems, financial amounts, and other significant entities. Explain each '
		             'entity only to the extent supported by the retrieved document excerpts.'),
		'Tables': ('Identify and extract material tables and tabular information from the active '
		           'document or documents. Preserve column meanings, row relationships, labels, '
		           'units, totals, and relevant explanatory context. State clearly when the '
		           'retrieved excerpts do not contain an extractable table.'),
		'Compare': ('Compare the active documents. Identify material similarities, differences, '
		            'conflicts, changes, omissions, and relationships. Organize the comparison by '
		            'topic and attribute each conclusion to the applicable document.'), }
	
	if task_value not in task_prompts:
		raise ValueError( f'Unsupported document task: {task_value}.' )
	
	task_prompt: str = task_prompts[ task_value ]
	
	if refinement_value:
		task_prompt = (f'{task_prompt}\n\n'
		               f'Additional user criteria:\n'
		               f'{refinement_value}')
	
	return task_prompt

def get_document_task_display_text( task: str, refinement: str, ) -> str:
	"""Document task display-text construction.

	Purpose:
		Creates the concise user-message text displayed in the Document Q&A
		conversation for a submitted task.

	Args:
		task (str): Selected Document Q&A task.
		refinement (str): Optional user-supplied task scope or criteria.

	Returns:
		str: User-facing conversation text.
	"""
	task_value: str = str( task or 'Ask a Question' ).strip( )
	
	refinement_value: str = str( refinement or '' ).strip( )
	
	if task_value == 'Ask a Question':
		return refinement_value
	
	if refinement_value:
		return f'{task_value}: {refinement_value}'
	
	return task_value

def invalidate_document_index_state( ) -> None:
	"""Document index-state invalidation.

	Purpose:
	    Invalidates Document Q&A vector-index metadata and clears stale retrieval
	    results after the uploaded or active document collection changes.

	Args:
	    None.

	Returns:
	    None: This function updates Document Q&A session state and does not return
	    a value.
	"""
	st.session_state[ 'docqna_fingerprint' ] = ''
	st.session_state[ 'docqna_vec_ready' ] = False
	st.session_state[ 'docqna_chunk_count' ] = 0
	st.session_state[ 'docqna_fallback_rows' ] = [ ]
	st.session_state[ 'doc_last_retrieval_hits' ] = [ ]

def get_document_uploader_key( ) -> str:
	"""Document uploader key retrieval.

	Purpose:
	    Returns the current revision-based Streamlit key used by the Document Q&A
	    file uploader.

	Args:
	    None.

	Returns:
	    str: Current Document Q&A file-uploader widget key.
	"""
	revision: int = int( st.session_state.get( 'doc_file_uploader_revision', 0, ) )
	return f'doc_file_uploader_{revision}'

def invalidate_document_index_state( ) -> None:
	"""Document index-state invalidation.

	Purpose:
	    Invalidates Document Q&A vector-index metadata and clears stale retrieval
	    results after the uploaded or active document collection changes.

	Args:
	    None.

	Returns:
	    None: This function updates Document Q&A session state and does not return
	    a value.
	"""
	st.session_state[ 'docqna_fingerprint' ] = ''
	st.session_state[ 'docqna_vec_ready' ] = False
	st.session_state[ 'docqna_chunk_count' ] = 0
	st.session_state[ 'docqna_fallback_rows' ] = [ ]
	st.session_state[ 'doc_last_retrieval_hits' ] = [ ]

def synchronize_uploaded_documents( uploader_key: str, ) -> None:
	"""Uploaded document synchronization.

	Purpose:
	    Synchronizes uploaded file objects, document bytes, and active-document
	    selections from the current revision-based file uploader.

	Args:
	    uploader_key (str): Session-state key owned by the current file uploader.

	Returns:
	    None: This function updates Document Q&A session state and does not return
	    a value.
	"""
	uploaded_value: Any = st.session_state.get( uploader_key, [ ], )
	uploaded_files: List[ Any ] = (
		list( uploaded_value ) if isinstance( uploaded_value, (tuple, list), ) else [ ])
	
	previous_uploaded_names: List[ str ] = [
		str( getattr( uploaded_file, 'name', '', ) or '' ).strip( ) for uploaded_file in
		st.session_state.get( 'uploaded', [ ], ) if
		str( getattr( uploaded_file, 'name', '', ) or '' ).strip( ) ]
	
	uploaded_names: List[ str ] = [ str( getattr( uploaded_file, 'name', '', ) or '' ).strip( ) for
		uploaded_file in uploaded_files if
		str( getattr( uploaded_file, 'name', '', ) or '' ).strip( ) ]
	
	document_bytes: Dict[ str, bytes ] = { }
	for uploaded_file in uploaded_files:
		document_name: str = str( getattr( uploaded_file, 'name', '', ) or '' ).strip( )
		if not document_name:
			continue
		
		try:
			file_bytes: bytes = uploaded_file.getvalue( )
		except Exception:
			continue
		
		if file_bytes:
			document_bytes[ document_name ] = file_bytes
	
	current_active_documents: List[ str ] = [ str( document_name ) for document_name in
		st.session_state.get( 'active_docs', [ ], ) if str( document_name ) in uploaded_names ]
	
	new_document_names: List[ str ] = [ document_name for document_name in uploaded_names if
		document_name not in previous_uploaded_names ]
	
	for document_name in new_document_names:
		if document_name not in current_active_documents:
			current_active_documents.append( document_name )
	
	if (not previous_uploaded_names and uploaded_names and not current_active_documents):
		current_active_documents = uploaded_names.copy( )
	
	st.session_state[ 'uploaded' ] = uploaded_files
	st.session_state[ 'doc_bytes' ] = document_bytes
	st.session_state[ 'active_docs' ] = current_active_documents
	invalidate_document_index_state( )
	st.session_state[ 'doc_inventory_rows' ] = (build_document_inventory_rows( ))

def handle_active_document_change( ) -> None:
	"""Active document change handling.

	Purpose:
	    Normalizes the active-document selection, refreshes the inventory, and
	    invalidates stale retrieval-index state.

	Args:
	    None.

	Returns:
	    None: This function updates Document Q&A session state and does not return
	    a value.
	"""
	uploaded_names: List[ str ] = [ str( getattr( uploaded_file, 'name', '', ) or '' ).strip( ) for
		uploaded_file in st.session_state.get( 'uploaded', [ ], ) if
		str( getattr( uploaded_file, 'name', '', ) or '' ).strip( ) ]
	
	active_documents: List[ str ] = [ str( document_name ) for document_name in
		st.session_state.get( 'active_docs', [ ], ) if str( document_name ) in uploaded_names ]
	
	st.session_state[ 'active_docs' ] = active_documents
	
	invalidate_document_index_state( )
	
	st.session_state[ 'doc_inventory_rows' ] = (build_document_inventory_rows( ))

def unload_documents( ) -> None:
	"""Document collection unloading.

	Purpose:
	    Clears uploaded documents and retrieval state, then advances the uploader
	    revision so Streamlit creates a new empty file-uploader widget.

	Args:
	    None.

	Returns:
	    None: This function updates Document Q&A session state and does not return
	    a value.
	"""
	st.session_state[ 'uploaded' ] = [ ]
	st.session_state[ 'active_docs' ] = [ ]
	st.session_state[ 'doc_bytes' ] = { }
	st.session_state[ 'doc_inventory_rows' ] = [ ]
	
	st.session_state[ 'doc_file_uploader_revision' ] = (
			int( st.session_state.get( 'doc_file_uploader_revision', 0, ) ) + 1)
	
	invalidate_document_index_state( )
	
# ----- Reset Callbacks

def reset_task_preset_controls( ) -> None:
			"""Task Preset control reset.

			Purpose:
			    Restores every control in the Task Preset expander to its declared
			    application default.

			Args:
			    None.

			Returns:
			    None: This function performs its work through side effects and does not
			    return a value.
			"""
			for control_key, default_value in TASK_PRESET_DEFAULTS.items( ):
				st.session_state[ control_key ] = default_value

def reset_reasoning_controls( ) -> None:
	"""Reasoning control reset.

	Purpose:
	    Restores every control in the Reasoning Controls expander to its declared
	    application default.

	Args:
	    None.

	Returns:
	    None: This function performs its work through side effects and does not
	    return a value.
	"""
	for control_key, default_value in REASONING_CONTROL_DEFAULTS.items( ):
		st.session_state[ control_key ] = default_value

def reset_coding_controls( ) -> None:
	"""Coding control reset.

	Purpose:
	    Restores every control in the Coding Controls expander to its declared
	    application default.

	Args:
	    None.

	Returns:
	    None: This function performs its work through side effects and does not
	    return a value.
	"""
	for control_key, default_value in CODING_CONTROL_DEFAULTS.items( ):
		st.session_state[ control_key ] = default_value

def reset_retrieval_controls( ) -> None:
	"""Retrieval control reset.

	Purpose:
	    Restores every control in the Retrieval Controls expander to its declared
	    application default.

	Args:
	    None.

	Returns:
	    None: This function performs its work through side effects and does not
	    return a value.
	"""
	for control_key, default_value in RETRIEVAL_CONTROL_DEFAULTS.items( ):
		st.session_state[ control_key ] = default_value

def apply_text_generation_preset_callback( ) -> None:
	"""Text Generation preset callback.

	Purpose:
	    Applies the selected Text Generation task preset, clears the selected stored
	    prompt, and closes the effective-prompt preview.

	Args:
	    None.

	Returns:
	    None: This function performs its work through side effects and does not
	    return a value.
	"""
	apply_text_generation_preset( )
	
	st.session_state[ 'text_instruction_prompt_id' ] = None
	st.session_state[ 'preview_effective_prompt' ] = False

def reset_text_generation_system_instructions( ) -> None:
	"""Text Generation System Instructions reset.

	Purpose:
	    Restores the Text Generation System Instructions controls to their initial
	    category and clears the selected prompt, editable instructions, prompt
	    metadata, and effective-prompt preview.

	Args:
	    None.

	Returns:
	    None: This function performs its work through side effects and does not
	    return a value.
	"""
	reset_system_instruction_controls( category_key='text_instruction_category',
		prompt_id_key='text_instruction_prompt_id',
		allowed_categories=TEXT_GENERATION_PROMPT_CATEGORIES, clear_text=True, )
	
	st.session_state[ 'preview_effective_prompt' ] = False

def toggle_text_generation_prompt_preview( ) -> None:
	"""Text Generation prompt-preview toggle.

	Purpose:
	    Toggles the visibility of the effective Text Generation prompt preview.

	Args:
	    None.

	Returns:
	    None: This function performs its work through side effects and does not
	    return a value.
	"""
	st.session_state[ 'preview_effective_prompt' ] = not bool(
		st.session_state.get( 'preview_effective_prompt', False, ) )

# ----- Life-Cycle Utilities -----

def begin_generation( provider: str = 'local', ) -> int:
	"""Begin a model-generation request.

	Purpose:
		Initializes the shared generation lifecycle for a new request, clears any
		previous cancellation request, records the active provider, and returns a
		unique request identifier.

	Args:
		provider: Provider executing the generation request.

	Returns:
		int: Unique identifier assigned to the new generation request.
	"""
	request_id: int = int( st.session_state.get( 'generation_request_id', 0, ) ) + 1
	
	st.session_state[ 'generation_request_id' ] = request_id
	st.session_state[ 'generation_active' ] = True
	st.session_state[ 'generation_stop_requested' ] = False
	st.session_state[ 'generation_provider' ] = str( provider or 'local' )
	st.session_state[ 'generation_status' ] = 'generating'
	
	return request_id

def request_generation_stop( ) -> None:
	"""Request cancellation of the active generation operation.

	Purpose:
		Sets the shared cancellation flag read by cancellation-aware streaming
		execution paths.

	Args:
		None.

	Returns:
		None: This function updates shared generation state.
	"""
	if bool( st.session_state.get( 'generation_active', False, ) ):
		st.session_state[ 'generation_stop_requested' ] = True
		st.session_state[ 'generation_status' ] = 'stopping'

def generation_stop_requested( request_id: int, ) -> bool:
	"""Determine whether an active generation request should stop.

	Purpose:
		Validates that the supplied request identifier is still current and returns
		the shared cancellation state for that request.

	Args:
		request_id: Identifier assigned when the generation operation began.

	Returns:
		bool: True when the current request has been cancelled or superseded;
		otherwise False.
	"""
	current_request_id: int = int( st.session_state.get( 'generation_request_id', 0, ) )
	
	if int( request_id ) != current_request_id:
		return True
	
	return bool( st.session_state.get( 'generation_stop_requested', False, ) )

def complete_generation( request_id: int, status: str = 'completed', ) -> None:
	"""Complete a model-generation request.

	Purpose:
		Finalizes the shared generation lifecycle when the supplied request remains
		the current active request.

	Args:
		request_id: Identifier assigned when the generation operation began.
		status: Final lifecycle status assigned to the request.

	Returns:
		None: This function updates shared generation state.
	"""
	current_request_id: int = int( st.session_state.get( 'generation_request_id', 0, ) )
	
	if int( request_id ) != current_request_id:
		return
	
	st.session_state[ 'generation_active' ] = False
	st.session_state[ 'generation_status' ] = str( status or 'completed' )
	
# -------------- LLM  UTILITIES -------------------

@st.cache_resource
def load_llm( ctx: int, threads: int, repeat_window: int=64 ) -> Any | None:
	"""Load the configured local language model.

	Purpose:
		Lazily loads and caches the configured llama.cpp model using the selected
		context-window size, CPU-thread count, and repetition-penalty window. Each
		distinct combination of runtime settings receives its own cached model
		instance.

	Args:
		ctx: Context-window size used to initialize the local model.
		threads: CPU-thread count used for local inference.
		repeat_window: Number of recent tokens considered when applying repetition,
		frequency, and presence penalties.

	Returns:
		Any | None: Initialized llama.cpp model instance when the model and dependency
		are available; otherwise None.
	"""
	try:
		if not local_model_available( ):
			return None
		
		from llama_cpp import Llama
		context_window_value: int = (int( ctx ) if int( ctx ) > 0 else int( cfg.DEFAULT_CTX ))
		cpu_thread_value: int = (int( threads ) if int( threads ) > 0 else int( cfg.CORES ))
		repeat_window_value: int = max( 0, int( repeat_window ), )
		return Llama( model_path=str( cfg.MODEL_PATH ), n_ctx=context_window_value,
			n_threads=cpu_thread_value, n_batch=512, last_n_tokens_size=repeat_window_value,
			verbose=False, )
	except Exception:
		return None
	
@st.cache_resource
def load_embedder( ) -> Any | None:
	"""
		Purpose:
		--------
		Lazily load the sentence embedding model when the dependency is available.

		Parameters:
		-----------
		None

		Returns:
		--------
		Any | None
			A sentence-transformer model instance when available; otherwise None.
	"""
	try:
		from sentence_transformers import SentenceTransformer
		
		return SentenceTransformer( 'all-MiniLM-L6-v2' )
	except Exception:
		return None

# ==============================================================================
# Init
# ==============================================================================
initialize_database( )
llm = None
embedder = None

if not isinstance( st.session_state.get( 'messages' ), list ):
	st.session_state[ 'messages' ] = [ ]

if len( st.session_state[ 'messages' ] ) == 0:
	st.session_state[ 'messages' ] = load_history( )

if 'system_instructions' not in st.session_state:
	st.session_state[ 'system_instructions' ] = ''

st.set_page_config( page_title=cfg.APP_TITLE, layout='wide', page_icon=cfg.FAVICON )
st.caption( cfg.APP_SUBTITLE )

# ==============================================================================
# SIDEBAR
# ==============================================================================
with st.sidebar:
	style_subheaders( )
	st.logo( cfg.LOGO_PATH, size='large' )
	
	c1, c2 = st.columns( [ 0.05, 0.95 ] )
	with c2:
		st.text( '⚙️ Application Mode' )
		st.markdown( cfg.BLUE_DIVIDER, unsafe_allow_html=True )
		mode = st.radio( label='', options=cfg.MODES, index=0 )
	
	st.divider( )

# ==============================================================================
# TEXT GENERATION MODE
# ==============================================================================
if mode == 'Text Generation':
	st.subheader( '💬 Text Generation', help=cfg.TEXT_GENERATION )
	st.divider( )
	
	messages = st.session_state.get( 'messages', [ ] )
	max_tokens = st.session_state.get( 'max_tokens', 0 )
	top_percent = st.session_state.get( 'top_percent', 0.0 )
	top_k = st.session_state.get( 'top_k', 0 )
	temperature = st.session_state.get( 'temperature', 0.0 )
	is_grounded = st.session_state.get( 'is_grounded', False )
	frequency_penalty = st.session_state.get( 'frequency_penalty', 0.0 )
	presense_penalty = st.session_state.get( 'presence_penalty', 0.0 )
	repeat_penalty = st.session_state.get( 'repeat_penalty', 0.0 )
	repeat_window = st.session_state.get( 'repeat_window', 0.0 )
	cpu_threads = st.session_state.get( 'cpu_threads', cfg.CORES )
	context_window = st.session_state.get( 'context_window', cfg.DEFAULT_CTX )
	# ----------------------------------------------------------------------------------
	# Main UI
	# ----------------------------------------------------------------------------------
	left, center, right = st.columns( [ 0.05, 0.9, 0.05 ] )
	with center:
		for control_key, default_value in { **TASK_PRESET_DEFAULTS, **REASONING_CONTROL_DEFAULTS,
			**CODING_CONTROL_DEFAULTS, }.items( ):
			if control_key not in st.session_state:
				st.session_state[ control_key ] = default_value
		
		# ----------------------------------------------------------------------------------
		# Expander - Mind Controls
		# ----------------------------------------------------------------------------------
		with st.expander( label='Mind Controls', icon='🧠', expanded=False, width='stretch', ):
			# ----------------------------------------------------------------------------------
			# Expander - TASK PRESET
			# ----------------------------------------------------------------------------------
			with st.expander( label='Task Preset', icon='🎯', expanded=False, ):
				task_c1, task_c2, task_c3, task_c4 = st.columns( [ 0.25, 0.25, 0.25, 0.25 ],
					border=True, gap='medium', )
				
				# ----- Task -----
				with task_c1:
					st.selectbox( label='Task', options=TASK_PRESET_OPTIONS, key='task_preset',
						help='Select the primary task performed by the model.', )
				
				# ----- Format -----
				with task_c2:
					st.selectbox( label='Response Format', options=RESPONSE_FORMAT_OPTIONS,
						key='response_format', help='Select the preferred response structure.', )
				
				# ----- History -----
				with task_c3:
					st.toggle( label='Use Chat History', key='use_chat_history',
						help='Include prior conversation turns in the effective prompt.', )
				
				# ----- Context -----
				with task_c4:
					st.toggle( label='Use Document Context', key='use_document_context',
						help='Include available document context in the effective prompt.', )
				
				# ----- Target -----
				st.text_input( label='Translation Target Language',
					key='translation_target_language',
					help=('Target language used when the Translation task preset is active.'), )
				
				# ----- Reset -----
				st.button( label='Reset', key='task_preset_reset', width='stretch', icon='🔄',
					on_click=reset_task_preset_controls, )
			
			# ----------------------------------------------------------------------------------
			# Expander - Reasoning Controls
			# ----------------------------------------------------------------------------------
			with st.expander( label='Reasoning Controls', icon='🧩', expanded=False, ):
				reason_c1, reason_c2, reason_c3, reason_c4 = st.columns( [ 0.25, 0.25, 0.25, 0.25 ],
					border=True, gap='medium', )
				
				# ----- Depth -----
				with reason_c1:
					st.selectbox( label='Reasoning Depth', options=REASONING_DEPTH_OPTIONS,
						key='reasoning_depth',
						help='Select the requested level of analytical depth.', )
				
				# ----- Answer -----
				with reason_c2:
					st.toggle( label='Answer Only', key='answer_only',
						help='Return the result without unnecessary introductory narration.', )
				
				# ----- Self-Check -----
				with reason_c3:
					st.toggle( label='Use Self-Check', key='use_self_check',
						help='Request verification of the conclusion before responding.', )
				
				# ----- Prefer -----
				with reason_c4:
					st.toggle( label='Prefer Deterministic Reasoning',
						key='deterministic_reasoning',
						help='Prefer stable and conservative reasoning behavior.', )
				
				# ----- Reset -----
				st.button( label='Reset', key='reasoning_controls_reset', width='stretch',
					on_click=reset_reasoning_controls, icon='🔄' )
			
			# ----------------------------------------------------------------------------------
			# Expander - Coding Controls
			# ----------------------------------------------------------------------------------
			with st.expander( label='Coding Controls', icon='🧾', expanded=False, ):
				code_c1, code_c2, code_c3, code_c4, code_c5 = st.columns(
					[ 0.20, 0.20, 0.20, 0.20, 0.20 ], border=True, gap='medium', )
				
				# ----- Language -----
				with code_c1:
					st.selectbox( label='Language / Technology',
						options=CODING_LANGUAGE_OPTIONS, key='coding_language',
						help=('Select the programming language, markup language, or '
						      'stylesheet technology for the coding task.'), )
				
				# ----- Task -----
				with code_c2:
					st.selectbox( label='Coding Task', options=CODING_TASK_OPTIONS,
						key='coding_task',
						help='Select the type of coding assistance required.', )
				
				# ----- Comment -----
				with code_c3:
					st.toggle( label='Include Comments',
						key='coding_include_comments',
						help='Include useful documentation and implementation comments.', )
				
				# ----- Format -----
				with code_c4:
					st.toggle( label='Use Editor Format',
						key='coding_editor_format',
						help='Return editor-ready source instead of pseudocode.', )
				
				# ----- Fencing -----
				with code_c5:
					st.toggle( label='Emit Fenced Code',
						key='coding_fenced_output',
						help='Wrap generated source in Markdown code fences.', )
				
				# ----- Reset -----
				st.button( label='Reset', key='coding_controls_reset',
					width='stretch', on_click=reset_coding_controls, icon='🔄', )
			
			# ----------------------------------------------------------------------------------
			# Expander - Response Controls
			# ----------------------------------------------------------------------------------
			with st.expander( label='Response Controls', icon='↔️',
					expanded=False, ):
				if int( st.session_state.get( 'max_tokens', 0, ) or 0 ) <= 0:
					st.session_state[ 'max_tokens' ] = 1024
				
				if float( st.session_state.get( 'top_percent', 0.0, ) or 0.0 ) <= 0.0:
					st.session_state[ 'top_percent' ] = 0.95
				
				if int( st.session_state.get( 'top_k', 0, ) or 0 ) <= 0:
					st.session_state[ 'top_k' ] = 40
				
				if float( st.session_state.get( 'repeat_penalty', 0.0, ) or 0.0 ) <= 0.0:
					st.session_state[ 'repeat_penalty' ] = 1.1
				
				if int( st.session_state.get( 'repeat_window', 0, ) or 0 ) <= 0:
					st.session_state[ 'repeat_window' ] = 64
				
				response_c1, response_c2, response_c3 = st.columns(
					[ 0.33, 0.34, 0.33 ], border=True, gap='medium', )
				
				# ----- Temperature -----
				with response_c1:
					st.slider( label='Temperature', min_value=0.0,
						max_value=2.0, step=0.01, key='temperature',
						help=('Controls sampling variation. Lower values produce more '
						      'stable responses; higher values increase variation.'), )
				
				# ----- Top-P -----
				with response_c2:
					st.slider( label='Top-P', min_value=0.0,
						max_value=1.0, step=0.01, key='top_percent',
						help=('Limits token selection to the smallest probability mass '
						      'meeting the selected threshold.'), )
				
				# ----- Top-K -----
				with response_c3:
					st.number_input( label='Top-K', min_value=0,
						max_value=1000, step=1, key='top_k',
						help=('Limits sampling to the selected number of highest-probability '
						      'tokens. Use zero only when supported by the model runtime.'), )
				
				response_c4, response_c5, response_c6 = st.columns(
					[ 0.33, 0.34, 0.33 ], border=True, gap='medium', )
				
				# ----- Frequency Penalty -----
				with response_c4:
					st.slider( label='Frequency Penalty', min_value=-2.0,
						max_value=2.0, step=0.01, key='frequency_penalty',
						help=('Adjusts the likelihood of tokens according to how often '
						      'they already appear in the generated response.'), )
				
				# ----- Presence Penalty -----
				with response_c5:
					st.slider( label='Presence Penalty', min_value=-2.0,
						max_value=2.0, step=0.01, key='presence_penalty',
						help=('Adjusts the likelihood of tokens according to whether '
						      'they have already appeared.'), )
				
				# ----- Repeat Penalty -----
				with response_c6:
					st.slider( label='Repeat Penalty', min_value=0.0,
						max_value=2.0, step=0.01, key='repeat_penalty',
						help=('Penalizes repeated token sequences during local-model '
						      'generation.'), )
				
				response_c7, response_c8, response_c9 = st.columns(
					[ 0.33, 0.34, 0.33 ], border=True, gap='medium', )
				
				# ----- Repeat Window -----
				with response_c7:
					st.number_input( label='Repeat Window', min_value=0,
						max_value=8192, step=1, key='repeat_window',
						help=('Number of recent local-model tokens considered when '
						      'applying repetition-related penalties.'), )
				
				# ----- Random Seed -----
				with response_c8:
					st.number_input( label='Random Seed', min_value=-1,
						max_value=2147483647, step=1, key='random_seed',
						help=('Controls repeatable local-model sampling. Use -1 for '
						      'runtime-selected nondeterministic behavior.'), )
				
				# ----- Maximum Tokens -----
				with response_c9:
					st.number_input( label='Maximum Tokens', min_value=1,
						max_value=32768, step=1, key='max_tokens',
						help='Maximum number of tokens generated for one response.', )
				
				st.markdown( cfg.BLUE_DIVIDER, unsafe_allow_html=True, )
				
				grounding_c1, grounding_c2 = st.columns(
					[ 0.5, 0.5 ], border=True, gap='medium', )
				
				# ----- Grounding -----
				with grounding_c1:
					st.toggle( label='Use Google Search Grounding',
						key='is_grounded',
						help=('Optionally route this Text Generation request through '
						      'Gemini with Google Search. Local inference remains the '
						      'default when this control is disabled.'), )
				
				# ----- Provider Status -----
				with grounding_c2:
					if bool( st.session_state.get( 'is_grounded', False, ) ):
						if gemini_grounding_available( ):
							st.success( 'Gemini grounding is configured.' )
						else:
							st.warning(
								'GEMINI_API_KEY is not configured. The request will '
								'use the local model.'
							)
					else:
						st.info( 'Provider: Local' )
				
				if bool( st.session_state.get( 'is_grounded', False, ) ):
					st.warning(
						'Grounded requests are sent to an external Gemini service. '
						'Do not submit sensitive, controlled, or private information.'
					)
					
					grounding_model_options: List[ str ] = (
						Chat( ).model_options
						or [ 'gemini-2.5-flash-lite' ]
					)
					
					current_grounding_model: str = str(
						st.session_state.get(
							'gemini_grounding_model',
							'gemini-2.5-flash-lite',
						)
					).strip( )
					
					if current_grounding_model not in grounding_model_options:
						st.session_state[ 'gemini_grounding_model' ] = (
							grounding_model_options[ 0 ]
						)
					
					st.selectbox( label='Grounding Model',
						options=grounding_model_options,
						key='gemini_grounding_model',
						help=('Select the Gemini model used only when optional '
						      'Google Search grounding is enabled.'), )
				
				# ----- Reset -----
				if st.button( label='Reset', key='response_controls_reset',
						width='stretch', icon='🔄', ):
					st.session_state[ 'temperature' ] = 0.0
					st.session_state[ 'top_percent' ] = 0.95
					st.session_state[ 'top_k' ] = 40
					st.session_state[ 'frequency_penalty' ] = 0.0
					st.session_state[ 'presence_penalty' ] = 0.0
					st.session_state[ 'repeat_penalty' ] = 1.1
					st.session_state[ 'repeat_window' ] = 64
					st.session_state[ 'random_seed' ] = -1
					st.session_state[ 'max_tokens' ] = 1024
					st.session_state[ 'is_grounded' ] = False
					st.session_state[ 'gemini_grounding_model' ] = (
						'gemini-2.5-flash-lite'
					)
					st.session_state[ 'gemini_grounding_error' ] = ''
					st.rerun( )
			
			# ----------------------------------------------------------------------------------
			# Expander - Inference Settings
			# ----------------------------------------------------------------------------------
			with st.expander( label='Inference Settings', icon='⚙️',
					expanded=False, ):
				if int( st.session_state.get( 'context_window', 0, ) or 0 ) <= 0:
					st.session_state[ 'context_window' ] = int( cfg.DEFAULT_CTX )
				
				if int( st.session_state.get( 'cpu_threads', 0, ) or 0 ) <= 0:
					st.session_state[ 'cpu_threads' ] = int( cfg.CORES )
				
				inference_c1, inference_c2, inference_c3 = st.columns(
					[ 0.33, 0.34, 0.33 ], border=True, gap='medium', )
				
				# ----- Context Window -----
				with inference_c1:
					st.number_input( label='Context Window', min_value=512,
						max_value=131072, step=512, key='context_window',
						help=('Maximum local-model context size used when loading '
						      'the configured GGUF model.'), )
				
				# ----- CPU Threads -----
				with inference_c2:
					st.number_input( label='CPU Threads', min_value=1,
						max_value=max( 1, int( cfg.CORES ) * 2, ), step=1,
						key='cpu_threads',
						help=('Number of CPU threads used by the local llama.cpp '
						      'inference runtime.'), )
				
				# ----- Local Model -----
				with inference_c3:
					if local_model_available( ):
						st.success( 'Local model available' )
					else:
						st.error( 'Local model unavailable' )
				
				st.caption( f'Configured Model: {cfg.MODEL_PATH}' )
				
				# ----- Reset -----
				if st.button( label='Reset', key='inference_settings_reset',
						width='stretch', icon='🔄', ):
					st.session_state[ 'context_window' ] = int( cfg.DEFAULT_CTX )
					st.session_state[ 'cpu_threads' ] = int( cfg.CORES )
					st.rerun( )
				 
		# ----------------------------------------------------------------------------------
		# Expander - System Instructions
		# ----------------------------------------------------------------------------------
		initialize_system_instruction_state( category_key='text_instruction_category',
			prompt_id_key='text_instruction_prompt_id',
			allowed_categories=TEXT_GENERATION_PROMPT_CATEGORIES, )
		
		with st.expander( label='System Instructions', icon='🖥️', expanded=False,
				width='stretch', ):
			available_categories: List[ str ] = get_available_prompt_categories(
				allowed_categories=TEXT_GENERATION_PROMPT_CATEGORIES, )
			
			selected_category: str = str(
				st.session_state.get( 'text_instruction_category', '', ) or '' ).strip( )
			
			prompt_ids, prompt_options = get_prompt_ids_for_category( category=selected_category, )
			selector_c1, selector_c2 = st.columns( [ 0.35, 0.65 ], border=True, gap='medium', )
			
			# ----- Category -----
			with selector_c1:
				st.selectbox( label='Category', options=available_categories,
					key='text_instruction_category', on_change=change_system_instruction_category,
					args=('text_instruction_category', 'text_instruction_prompt_id',),
					disabled=len( available_categories ) == 0,
					help='Select a prompt category available to Text Generation mode.', )
			
			# ----- Template -----
			with selector_c2:
				st.selectbox( label='Prompt Template', options=prompt_ids,
					key='text_instruction_prompt_id', index=None,
					format_func=lambda prompt_id: format_prompt_option( prompt_id=prompt_id,
						prompt_options=prompt_options, ),
					on_change=load_selected_prompt_into_system_instructions,
					args=('text_instruction_prompt_id',), disabled=len( prompt_ids ) == 0,
					help='Select a prompt template by its stored Prompts.ID value.', )
			
			# ------ Edit -----
			st.text_area( label='Enter Text', height=180, width='stretch',
				help=cfg.SYSTEM_INSTRUCTIONS, key='system_instructions', )
			
			active_prompt_caption: str = str(
				st.session_state.get( 'active_prompt_caption', '', ) or '' ).strip( )
			
			active_prompt_name: str = str(
				st.session_state.get( 'active_prompt_name', '', ) or '' ).strip( )
			
			selected_prompt_id: Any = st.session_state.get( 'selected_prompt_id', )
			if selected_prompt_id and active_prompt_caption:
				prompt_metadata_parts: List[ str ] = [ f'ID: {int( selected_prompt_id )}',
						f'Caption: {active_prompt_caption}', ]
				
				st.caption( ' | '.join( prompt_metadata_parts ) )
			
			# ----- Actions -----
			user_preview_input: str = str( st.session_state.get( 'last_preview_input', '',
			) or '' )
			
			btn_c1, btn_c2, btn_c3, btn_c4 = st.columns( [ 0.25, 0.25, 0.25, 0.25 ], )
			
			# ----- Clear Instructions -----
			with btn_c1:
				st.button( label='Clear Instructions', key='text_clear_instructions',
					width='stretch', on_click=clear_system_instruction_selection, icon='🧹',
					kwargs={ 'prompt_id_key': 'text_instruction_prompt_id', 'clear_text': True,}, )
			
			# ----- Convert Format -----
			with btn_c2:
				st.button( label='XML ↔️ Markdown', key='text_convert_instructions',
					width='stretch', on_click=convert_system_instruction_text, )
			
			# ----- Apply Preset -----
			with btn_c3:
				st.button( label='Apply Preset', key='text_apply_preset', width='stretch',
					on_click=apply_text_generation_preset_callback, )
			
			# ----- Preview Prompt -----
			with btn_c4:
				st.button( label='Preview Prompt', key='text_preview_prompt', width='stretch',
					on_click=toggle_text_generation_prompt_preview, )
			
			# ----- Preview -----
			if bool( st.session_state.get( 'preview_effective_prompt', False, ) ):
				st.text_area( label='Effective Prompt Preview',
					value=build_effective_prompt_preview( user_preview_input, ), height=220,
					disabled=True, key='text_effective_prompt_preview', )
			
			# ----- Reset -----
			st.button( label='Reset', key='text_system_instruction_reset', width='stretch',
				on_click=reset_text_generation_system_instructions, icon='🔄' )
			
		st.markdown( cfg.BLUE_DIVIDER, unsafe_allow_html=True )
		
		# ----------------------------------------------------------------------------------
		# Messages
		# ----------------------------------------------------------------------------------
		for message in st.session_state.get( 'messages', [ ], ):
			message_role: str = ''
			message_content: str = ''
			
			if isinstance( message, dict, ):
				message_role = str( message.get( 'role', '', ) or '' ).strip( )
				message_content = str( message.get( 'content', '', ) or '' )
			
			elif isinstance( message, (tuple, list), ):
				if len( message ) == 2:
					message_role = str( message[ 0 ] or '' ).strip( )
					message_content = str( message[ 1 ] or '' )
			
			if message_role not in ('user', 'assistant', 'system',):
				continue
			
			if not message_content.strip( ):
				continue
			
			with st.chat_message( message_role, ):
				st.markdown( message_content )
		
		user_input: str | None = st.chat_input( 'Ask Jimi…', key='text_generation_chat_input', )
		
		if user_input:
			user_input_value: str = str( user_input ).strip( )
			
			if user_input_value:
				st.session_state[ 'last_preview_input' ] = (user_input_value)
				
				with st.chat_message( 'user', ):
					st.markdown( user_input_value )
				
				try:
					with st.chat_message( 'assistant', ):
						output = st.empty( )
						response_text: str = run_llm_turn( user_input=user_input_value,
							temperature=float( st.session_state.get( 'temperature', 0.0, ) ),
							top_p=float( st.session_state.get( 'top_percent', 0.95, ) ),
							repeat_penalty=float( st.session_state.get( 'repeat_penalty', 1.1, ) ),
							max_tokens=(int( st.session_state.get( 'max_tokens', 1024, ) ) or
							            1024),
							stream=True, output=output, )
					
					response_value: str = str( response_text or '' ).strip( )
					if not response_value:
						raise ValueError( 'No Text Generation response was returned.' )
					
					save_message( 'user', user_input_value, )
					save_message( 'assistant', response_value, )
					st.session_state[ 'messages' ].append( ('user', user_input_value,) )
					st.session_state[ 'messages' ].append( ('assistant', response_value,) )
				
				except ValueError as ex:
					st.error( str( ex ) )
				
				except Exception as ex:
					st.error( f'Text Generation failed: {ex}' )
		
		# ----------------------------------------------------------------------------------
		# Clear Chat
		# ----------------------------------------------------------------------------------
		if st.button( label='🧹 Clear Chat', key='text_clear_chat', width='stretch', ):
			clear_history( )
			st.session_state[ 'messages' ] = [ ]
			st.rerun( )

# ==============================================================================
# RETRIEVAL AUGMENTATION
# ==============================================================================
elif mode == 'Document Q&A':
	st.subheader( '📚 Retrieval Augementation', help=cfg.RETRIEVAL_AUGMENTATION )
	st.divider( )
	
	messages = st.session_state.get( 'messages', [ ] )
	uploaded = st.session_state.get( 'uploaded', [ ] )
	active_docs = st.session_state.get( 'active_docs', [ ] )
	doc_bytes = st.session_state.get( 'doc_bytes', { } )
	max_tokens = st.session_state.get( 'max_tokens', 0 )
	top_percent = st.session_state.get( 'top_percent', 0.0 )
	top_k = st.session_state.get( 'top_k', 0 )
	temperature = st.session_state.get( 'temperature', 0.0 )
	frequency_penalty = st.session_state.get( 'frequency_penalty', 0.0 )
	presence_penalty = st.session_state.get( 'presence_penalty', 0.0 )
	repeat_penalty = st.session_state.get( 'repeat_penalty', 0.0 )
	repeat_window = st.session_state.get( 'repeat_window', 0.0 )
	cpu_threads = st.session_state.get( 'cpu_threads', cfg.CORES )
	context_window = st.session_state.get( 'context_window', cfg.DEFAULT_CTX )
	
	# ----------------------------------------------------------------------------------
	# Main UI
	# ----------------------------------------------------------------------------------
	left, center, right = st.columns( [ 0.05, 0.9, 0.05 ] )
	with center:
		for control_key, default_value in RETRIEVAL_CONTROL_DEFAULTS.items( ):
			if control_key not in st.session_state:
				st.session_state[ control_key ] = default_value

		# -----------------------------------------------------------------------------------
		# Expander — Retrieval Controls
		# -----------------------------------------------------------------------------------
		with st.expander( label='Retrieval Controls', icon='🔎', expanded=False, ):
			retrieval_c1, retrieval_c2, retrieval_c3 = st.columns( [ 0.33, 0.34, 0.33 ],
				border=True, gap='medium', )
			
			# ----- Number -----
			with retrieval_c1:
				st.number_input( label='Retrieved Chunks', min_value=1, max_value=50, step=1,
					key='retrieval_k',
					help=('Maximum number of document chunks retrieved for each '
					      'question or document task.'), )
			
			# ----- Size -----
			with retrieval_c2:
				st.number_input( label='Chunk Size', min_value=200, max_value=8000, step=100,
					key='retrieval_chunk_size',
					help=('Maximum number of characters included in each indexed '
					      'document chunk.'), )
			
			# ----- Overlap -----
			with retrieval_c3:
				st.number_input( label='Chunk Overlap', min_value=0, max_value=2000, step=50,
					key='retrieval_chunk_overlap',
					help=('Number of characters shared by adjacent document chunks.'), )
			
			retrieval_c4, retrieval_c5, retrieval_c6 = st.columns( [ 0.33, 0.34, 0.33 ],
				border=True, gap='medium', )
			
			# ----- Show -----
			with retrieval_c4:
				st.toggle( label='Show Retrieved Chunks', key='show_retrieved_chunks',
					help=('Display the retrieved document excerpts used to construct '
					      'the response.'), )
			
			# ----- Grounded -----
			with retrieval_c5:
				st.toggle( label='Require Grounding', key='require_grounding',
					help=('Require responses to remain grounded in the active '
					      'document collection.'), )
			
			# ----- Experpts -----
			with retrieval_c6:
				st.toggle( label='Answer From Excerpts Only', key='answer_from_excerpts_only',
					help=('Restrict answers to facts supported by retrieved document '
					      'excerpts.'), )
			
			retrieval_c7, retrieval_c8 = st.columns( [ 0.5, 0.5 ], border=True, gap='medium', )
			
			# ----- SQLite -----
			with retrieval_c7:
				st.toggle( label='Prefer SQLite Vector Search', key='prefer_sqlite_vec',
					help=('Use the SQLite vector index as the preferred retrieval '
					      'backend when it is available.'), )
			
			# ----- Fallback -----
			with retrieval_c8:
				st.toggle( label='Allow Similarity Fallback', key='allow_similarity_fallback',
					help=('Use the fallback similarity-search implementation when '
					      'the preferred vector backend is unavailable.'), )
			
			# ----- Chunks -----
			if (int( st.session_state.get( 'retrieval_chunk_overlap', 0, ) ) >= int(
				st.session_state.get( 'retrieval_chunk_size', 1, ) )):
				st.warning( 'Chunk Overlap must be smaller than Chunk Size.' )
			
			# ----- Reset -----
			st.button( label='Reset', key='doc_retrieval_controls_reset', width='stretch',
				on_click=reset_retrieval_controls, icon='🔄' )
		
		# ----------------------------------------------------------------------------------
		# Expander - System Instructions
		# ----------------------------------------------------------------------------------
		initialize_system_instruction_state( category_key='doc_instruction_category',
			prompt_id_key='doc_instruction_prompt_id',
			allowed_categories=DOCUMENT_QNA_PROMPT_CATEGORIES, )
		
		with st.expander( label='System Instructions', icon='🖥️', expanded=False, width='stretch', ):
			available_doc_categories: List[ str ] = get_available_prompt_categories(
				allowed_categories=DOCUMENT_QNA_PROMPT_CATEGORIES, )
			
			selected_doc_category: str = str(
				st.session_state.get( 'doc_instruction_category', '', ) or '' ).strip( )
			
			doc_prompt_ids, doc_prompt_options = get_prompt_ids_for_category(
				category=selected_doc_category, )
			
			selector_c1, selector_c2 = st.columns( [ 0.35, 0.65 ], border=True, gap='medium', )
			
			# ----- Category -----
			with selector_c1:
				st.selectbox( label='Category', options=available_doc_categories,
					key='doc_instruction_category', on_change=change_system_instruction_category,
					args=('doc_instruction_category', 'doc_instruction_prompt_id',),
					disabled=len( available_doc_categories ) == 0,
					help=('Select a prompt category available to Document Q&A mode.'), )
			
			# ----- Prompt Template -----
			with selector_c2:
				st.selectbox( label='Prompt Template', options=doc_prompt_ids,
					key='doc_instruction_prompt_id', index=None,
					format_func=lambda prompt_id: format_prompt_option( prompt_id=prompt_id,
						prompt_options=doc_prompt_options, ),
					on_change=load_selected_prompt_into_system_instructions,
					args=('doc_instruction_prompt_id',), disabled=len( doc_prompt_ids ) == 0,
					help=('Select a stored prompt template for the Document Q&A '
					      'System Instructions.'), )
			
			# ----- Editable Instructions -----
			st.text_area( label='Enter Text', height=180, width='stretch',
				help=cfg.SYSTEM_INSTRUCTIONS, key='system_instructions', )
			
			# ----- Active Prompt Metadata -----
			active_doc_prompt_caption: str = str(
				st.session_state.get( 'active_prompt_caption', '', ) or '' ).strip( )
			
			active_doc_prompt_id: Any = st.session_state.get( 'selected_prompt_id', )
			if (active_doc_prompt_id and active_doc_prompt_caption):
				doc_prompt_metadata_parts: List[ str ] = [ f'ID: {int( active_doc_prompt_id )}',
					f'Caption: {active_doc_prompt_caption}', ]
				
				st.caption( ' | '.join( doc_prompt_metadata_parts ) )
			
			# ----- Actions -----
			action_c1, action_c2 = st.columns( [ 0.5, 0.5 ], gap='medium', )
			
			# ----- Clear Instructions -----
			with action_c1:
				st.button( label='Clear Instructions', key='doc_clear_instructions',
					width='stretch', on_click=clear_system_instruction_selection, icon='🧹',
					kwargs={ 'prompt_id_key': 'doc_instruction_prompt_id', 'clear_text': True, }, )
			
			# ----- Convert Format -----
			with action_c2:
				st.button( label='XML ↔️ Markdown', key='doc_convert_instructions',
					width='stretch', on_click=convert_system_instruction_text, )
			
			# ----- Reset -----
			st.button( label='Reset', key='doc_system_instruction_reset', width='stretch',
				on_click=reset_system_instruction_controls, icon='🔄',
				kwargs={ 'category_key': 'doc_instruction_category',
					'prompt_id_key': 'doc_instruction_prompt_id',
					'allowed_categories': DOCUMENT_QNA_PROMPT_CATEGORIES, 'clear_text': True, }, )
			
		# ----------------------------------------------------------------------------------
		# Expander - Document Loader
		# ----------------------------------------------------------------------------------
		with st.expander( label='Document Loader', icon='📥', expanded=False, width='stretch', ):
			doc_left, doc_right = st.columns( [ 0.5, 0.5 ], gap='medium', border=True, )
			
			# ----- SOURCE AND SELECTION ------
			with doc_left:
				st.radio( label='Document Source', options=[ 'uploadlocal' ], index=0,
					horizontal=True, key='doc_source', )
				
				document_uploader_key: str = get_document_uploader_key( )
				st.file_uploader( label='Upload document(s) (PDF, TXT, DOCX)',
					type=[ 'pdf', 'txt', 'docx', ], accept_multiple_files=True,
					label_visibility='visible', key=document_uploader_key,
					on_change=synchronize_uploaded_documents, args=(document_uploader_key,), )
				
				uploaded_files: List[ Any ] = (st.session_state.get( 'uploaded', [ ], ))
				uploaded_names: List[ str ] = [
					str( getattr( uploaded_file, 'name', '', ) or '' ).strip( ) for
					uploaded_file in
					uploaded_files if str( getattr( uploaded_file, 'name', '', ) or '' ).strip( ) ]
				
				if uploaded_names:
					current_active_documents: List[ str ] = [ str( document_name ) for
						document_name
						in st.session_state.get( 'active_docs', [ ], ) if
						str( document_name ) in uploaded_names ]
					
					if (current_active_documents != st.session_state.get( 'active_docs', [ ], )):
						st.session_state[ 'active_docs' ] = (current_active_documents)
					
					st.multiselect( label='Active Documents', options=uploaded_names,
						key='active_docs', on_change=handle_active_document_change,
						help=('Select the uploaded documents included in retrieval, '
						      'comparison, preview, and task execution.'), )
				
				else:
					st.info( 'Load one or more documents.' )
				
				st.button( label='Unload Document(s)', key='doc_unload_documents', width='stretch',
					disabled=len( uploaded_names ) == 0, on_click=unload_documents, )
				
				if bool( st.session_state.get( 'show_doc_parse_diagnostics', False, ) ):
					st.caption( f'Chunk Size: '
					            f'{int( st.session_state.get( "retrieval_chunk_size", 1200 ) )} '
					            f'| Chunk Overlap: '
					            f'{int( st.session_state.get( "retrieval_chunk_overlap", 200 ) )} '
					            f'| Index Ready: '
					            f'{bool( st.session_state.get( "docqna_vec_ready", False ) )} '
					            f'| Chunk Count: '
					            f'{int( st.session_state.get( "docqna_chunk_count", 0 ) )}' )
			
			# ----- DOCUMENT PREVIEW ------
			with doc_right:
				active_documents: List[ str ] = [ str( document_name ) for document_name in
					st.session_state.get( 'active_docs', [ ], ) if
					str( document_name or '' ).strip( ) ]
				
				document_bytes: Dict[ str, bytes ] = (st.session_state.get( 'doc_bytes', { }, ))
				
				if active_documents:
					preview_name: str = active_documents[ 0 ]
					preview_bytes: bytes | None = document_bytes.get( preview_name )
					if (preview_bytes and preview_name.lower( ).endswith( '.pdf' )):
						st.pdf( preview_bytes, height=420, )
					
					elif preview_bytes:
						preview_text: str = extract_text( file_bytes=preview_bytes,
							file_name=preview_name, )
						
						if preview_text:
							st.text_area( label=f'Preview: {preview_name}',
								value=preview_text[ :4000 ], height=420, disabled=True,
								key='doc_preview_text', )
						
						else:
							st.info( 'The selected document did not contain extractable text.' )
					
					else:
						st.info( 'Document loaded but preview unavailable.' )
				
				elif uploaded_names:
					st.info( 'Select at least one active document.' )
				
				else:
					st.info( 'No document loaded.' )
			
			# ----- ACTIVE DOCUMENT INVENTORY ------
			inventory_rows: List[ Dict[ str, Any ] ] = (
				st.session_state.get( 'doc_inventory_rows', [ ], ))
			
			if inventory_rows:
				st.markdown( '#### Active Document Inventory' )
				
				st.dataframe( pd.DataFrame( inventory_rows ), use_container_width=True, )
		
		# ------------------------------------------------------------------
		# Messages
		# ------------------------------------------------------------------
		if 'messages' not in st.session_state:
			st.session_state[ 'messages' ] = [ ]
		
		if not isinstance( st.session_state.get( 'messages' ), list, ):
			st.session_state[ 'messages' ] = [ ]
		
		if 'docqna_task' not in st.session_state:
			st.session_state[ 'docqna_task' ] = 'Ask a Question'
		
		for message in st.session_state[ 'messages' ]:
			message_role: str = ''
			message_content: str = ''
			
			if isinstance( message, dict, ):
				message_role = str( message.get( 'role', '', ) or '' ).strip( )
				
				message_content = str( message.get( 'content', '', ) or '' )
			
			elif isinstance( message, (tuple, list), ):
				if len( message ) == 2:
					message_role = str( message[ 0 ] or '' ).strip( )
					
					message_content = str( message[ 1 ] or '' )
			
			if message_role not in ('user', 'assistant', 'system',):
				continue
			
			if not message_content.strip( ):
				continue
			
			with st.chat_message( message_role, ):
				st.markdown( message_content )
		
		st.markdown( cfg.BLUE_DIVIDER, unsafe_allow_html=True, )
		
		# ----- Task Selection ------
		selected_document_task: str = st.radio( label='Document Task',
			options=DOCUMENT_TASK_OPTIONS, key='docqna_task', horizontal=True,
			help=('Select how the next submission should process the active documents.'), )
		
		st.caption( DOCUMENT_TASK_CAPTIONS.get( selected_document_task,
			DOCUMENT_TASK_CAPTIONS[ 'Ask a Question' ], ) )
		
		active_document_names: List[ str ] = [ str( document_name ) for document_name in
			st.session_state.get( 'active_docs', [ ], ) if str( document_name or '' ).strip( ) ]
		
		has_active_documents: bool = (len( active_document_names ) > 0)
		compare_available: bool = (len( active_document_names ) >= 2)
		if not has_active_documents:
			st.warning(
				'Load and activate at least one document before submitting a document task.' )
		
		elif (selected_document_task == 'Compare' and not compare_available):
			st.warning( 'Compare requires at least two active documents.' )
		
		# ----- Task Execution -----
		run_default_task: bool = False
		if selected_document_task != 'Ask a Question':
			run_default_task = st.button( label='Run Task', key='docqna_run_selected_task',
				width='stretch', disabled=(not has_active_documents or (
						selected_document_task == 'Compare' and not compare_available)),
				help=('Runs the selected task using its default instructions. Use the chat '
				      'input to provide optional task-specific criteria.'), )
		
		# ----- Task Input -----
		chat_placeholder: str = DOCUMENT_TASK_PLACEHOLDERS.get( selected_document_task,
			DOCUMENT_TASK_PLACEHOLDERS[ 'Ask a Question' ], )
		
		chat_submission: str | None = st.chat_input( placeholder=chat_placeholder,
			key='docqna_chat_input', disabled=(not has_active_documents or (
					selected_document_task == 'Compare' and not compare_available)), )
		
		refinement_text: str = (
			str( chat_submission ).strip( ) if chat_submission is not None else '')
		
		execute_document_task: bool = (run_default_task or bool( refinement_text ))
		
		# ----- Task Routing -----
		if execute_document_task:
			try:
				if not has_active_documents:
					raise ValueError(
						'Load and activate at least one document before submitting a task.' )
				
				if (selected_document_task == 'Compare' and not compare_available):
					raise ValueError( 'Compare requires at least two active documents.' )
				
				execution_prompt: str = build_document_task_prompt( task=selected_document_task,
					refinement=refinement_text, )
				
				display_text: str = get_document_task_display_text( task=selected_document_task,
					refinement=refinement_text, )
				
				if not display_text:
					raise ValueError( 'Enter a document question before submitting.' )
				
				# Render the task without inserting it into prior conversation history.
				with st.chat_message( 'user' ):
					st.markdown( display_text )
				
				document_prompt: str = build_document_user_input( user_query=execution_prompt,
					k=int( st.session_state.get( 'retrieval_k', 6, ) ), )
				
				if not document_prompt:
					raise ValueError( 'The Document Q&A prompt could not be constructed.' )
				
				with st.chat_message( 'assistant' ):
					output = st.empty( )
					
					response: str = run_model_prompt( prompt=document_prompt,
						temperature=float( st.session_state.get( 'temperature', 0.0, ) ),
						top_p=float( st.session_state.get( 'top_percent', 0.95, ) ),
						repeat_penalty=float( st.session_state.get( 'repeat_penalty', 1.1, ) ),
						max_tokens=(int( st.session_state.get( 'max_tokens', 1024, ) ) or 1024),
						stream=True, output=output, )
				
				response_text: str = str( response or '' ).strip( )
				
				if not response_text:
					raise ValueError( 'No Document Q&A response was returned.' )
				
				# Persist both messages only after prompt construction and generation.
				save_message( 'user', display_text, )
				save_message( 'assistant', response_text, )
				st.session_state[ 'messages' ].append( ('user', display_text,) )
				st.session_state[ 'messages' ].append( ('assistant', response_text,) )
			
			except ValueError as ex:
				st.error( str( ex ) )
			
			except Exception as ex:
				st.error( f'Document Q&A failed: {ex}' )
		
		# ----- Retrieved Chunks ------
		if bool( st.session_state.get( 'show_retrieved_chunks', True, ) ):
			retrieval_hits: List[ Tuple[ str, str, float ] ] = (
				st.session_state.get( 'doc_last_retrieval_hits', [ ], ))
			
			if retrieval_hits:
				with st.expander( label='Retrieved Chunks', expanded=False, ):
					for hit_index, retrieval_hit in enumerate( retrieval_hits, start=1, ):
						document_name: str = str( retrieval_hit[ 0 ] )
						chunk_value: str = str( retrieval_hit[ 1 ] )
						score_value: Any = retrieval_hit[ 2 ]
						st.markdown( f'**{hit_index}. {document_name}**' )
						st.caption( f'Score / Distance: {score_value}' )
						st.text_area( label=f'Chunk {hit_index}', value=chunk_value, height=140,
							disabled=True, key=f'doc_hit_{hit_index}', )
		
		# ----- Reset -----
		if st.button( label='🧹 Clear Chat', key='doc_clear_chat', width='stretch', ):
			clear_history( )
			st.session_state[ 'messages' ] = [ ]
			st.session_state[ 'doc_last_retrieval_hits' ] = [ ]
			st.rerun( )

# ==============================================================================
# SEMANTIC SEARCH
# ==============================================================================
elif mode == 'Semantic Search':
	st.subheader( '🔍 Semantic Search', help=cfg.SEMANTIC_SEARCH )
	st.divider( )
	
	# ----------------------------------------------------------------------------------
	# Main UI
	# ----------------------------------------------------------------------------------
	left, center, right = st.columns( [ 0.05, 0.9, 0.05 ] )
	with center:
		# ------------------------------------------------------------------
		# Expander - Index Builder
		# ------------------------------------------------------------------
		with st.expander( label='Index Builder', icon='🧱', expanded=False ):
			idx_c1, idx_c2, idx_c3, idx_c4 = st.columns( [ 0.25, 0.25, 0.25, 0.25 ], border=True,
				gap='medium' )
			
			# ----- Size -----
			with idx_c1:
				st.slider( label='Chunk Size', min_value=256, max_value=4000, step=64,
					key='semantic_chunk_size' )
			
			# ----- Overlap -----
			with idx_c2:
				st.slider( label='Chunk Overlap', min_value=0, max_value=1000, step=25,
					key='semantic_chunk_overlap' )
			
			# ----- Clear -----
			with idx_c3:
				st.toggle( label='Clear Existing Index',
					value=bool( st.session_state.get( 'semantic_clear_existing', True ) ),
					key='semantic_clear_existing' )
			
			# ----- Append -----
			with idx_c4:
				st.toggle( label='Append to Existing Index',
					value=bool( st.session_state.get( 'semantic_append_existing', False ) ),
					key='semantic_append_existing' )
			
			# ----- Diagnostics -----
			st.toggle( label='Show Embedding Diagnostics',
				value=bool( st.session_state.get( 'semantic_show_diagnostics', True ) ),
				key='semantic_show_diagnostics' )
			
			# ----- Upload -----
			semantic_files = st.file_uploader( label='Upload for embedding',
				accept_multiple_files=True, type=[ 'pdf', 'txt', 'docx' ],
				key='semantic_file_uploader' )
			
			# ----- Build -----
			if st.button( 'Build Index', key='semantic_build_index', width='stretch' ):
				if semantic_files:
					result = build_semantic_index( semantic_files )
					if bool( result.get( 'success', False ) ):
						st.success( str( result.get( 'message', '' ) ) )
					else:
						st.error( str( result.get( 'message', 'Index build failed.' ) ) )
				else:
					st.info( 'Upload one or more files before building the index.' )
			
			# ----- Diagnostic -----
			if bool( st.session_state.get( 'semantic_show_diagnostics', True ) ):
				diag_c1, diag_c2, diag_c3 = st.columns( [ 0.33, 0.33, 0.34 ] )
				
				# ----- Documents -----
				with diag_c1:
					st.metric( 'Indexed Documents',
						int( st.session_state.get( 'semantic_index_doc_count', 0 ) ) )
				
				# ----- Chunks -----
				with diag_c2:
					st.metric( 'Indexed Chunks',
						int( st.session_state.get( 'semantic_index_chunk_count', 0 ) ) )
				
				# ----- Dimensions -----
				with diag_c3:
					st.metric( 'Vector Dimension',
						int( st.session_state.get( 'semantic_index_dim', 0 ) ) )
		
		# ------------------------------------------------------------------
		# Expander - Semantic Query
		# ------------------------------------------------------------------
		with st.expander( label='Semantic Query', icon='🧠', expanded=False ):
			query_c1, query_c2, query_c3 = st.columns( [ 0.34, 0.33, 0.33 ], border=True,
				gap='medium' )
			
			# ----- Top-K -----
			with query_c1:
				st.slider( label='Top K', min_value=1, max_value=25, step=1, key='semantic_top_k' )
			
			# ----- Similarity -----
			with query_c2:
				st.slider( label='Minimum Similarity', min_value=0.0, max_value=1.0, step=0.01,
					key='semantic_min_similarity' )
			
			# ----- Grouping -----
			with query_c3:
				st.toggle( label='Group by Document',
					value=bool( st.session_state.get( 'semantic_group_by_document', False ) ),
					key='semantic_group_by_document' )
			
			# ----- Query -----
			semantic_query = st.text_area( label='Semantic Query', height=120,
				key='semantic_query_text' )
			
			# ----- Run -----
			if st.button( 'Run Semantic Search', key='semantic_run_query', width='stretch' ):
				rows = query_semantic_index( semantic_query )
				if len( rows ) == 0:
					st.info( 'No semantic matches found.' )
			
			result_rows = st.session_state.get( 'semantic_result_rows', [ ] )
			if isinstance( result_rows, list ) and len( result_rows ) > 0:
				edited_rows = st.data_editor( result_rows, hide_index=True,
					use_container_width=True, key='semantic_results_editor' )
				
				selected_rows = extract_selected_semantic_rows( edited_rows )
				st.session_state[ 'semantic_selected_rows' ] = selected_rows
				if len( selected_rows ) > 0:
					st.caption( f'Selected Chunks: {len( selected_rows )}' )
		
		# ------------------------------------------------------------------
		# Expander - Actions
		# ------------------------------------------------------------------
		with st.expander( label='Actions', icon='🔀', expanded=False ):
			act_c1, act_c2, act_c3 = st.columns( [ 0.34, 0.33, 0.33 ], border=True )
			
			# ----- To Text -----
			with act_c1:
				if st.button( 'Send Selected Chunks to Text Generation', width='stretch' ):
					send_selected_semantic_chunks_to_text_generation( )
					st.success( 'Selected chunks added to shared Text Generation context.' )
			
			# ----- To Documents -----
			with act_c2:
				if st.button( 'Send Selected Chunks to Document Q&A', width='stretch' ):
					send_selected_semantic_chunks_to_doc_qna( )
					st.success( 'Selected chunks added to the shared semantic context buffer.' )
			
			# ----- To Context -----
			with act_c3:
				if st.button( 'Save Selected Chunks as Prompt Context', width='stretch' ):
					context_text = build_semantic_context_from_selection( )
					if context_text:
						existing_docs = st.session_state.get( 'basic_docs', [ ] )
						if not isinstance( existing_docs, list ):
							existing_docs = [ ]
						existing_docs.append( context_text )
						st.session_state[ 'basic_docs' ] = existing_docs
						st.success( 'Selected chunks saved to shared prompt context.' )
					else:
						st.info( 'Select one or more chunks first.' )
			
			selected_rows = st.session_state.get( 'semantic_selected_rows', [ ] )
			if isinstance( selected_rows, list ) and len( selected_rows ) > 0:
				st.markdown( '### Selected Semantic Context Preview' )
				st.text_area( label='Selected Context',
					value=build_semantic_context_from_selection( ), height=220, disabled=True )
		
		# ------------------------------------------------------------------
		# Expander - Index Maintenance
		# ------------------------------------------------------------------
		with st.expander( label='Index Maintenance', icon='🛠️', expanded=False ):
			maint_c1, maint_c2, maint_c3 = st.columns( [ 0.34, 0.33, 0.33 ], border=True )
			
			# ----- Delete -----
			with maint_c1:
				if st.button( 'Delete Index', width='stretch' ):
					clear_semantic_index( )
					st.success( 'Semantic index deleted.' )
			
			# ----- Recompute -----
			with maint_c2:
				if st.button( 'Recompute Diagnostics', width='stretch' ):
					rows = decode_embedding_rows( )
					st.session_state[ 'semantic_index_chunk_count' ] = len( rows )
					if len( rows ) > 0:
						st.session_state[ 'semantic_index_dim' ] = int( rows[ 0 ][ 1 ].shape[ 0 ] )
					else:
						st.session_state[ 'semantic_index_dim' ] = 0
					st.success( 'Diagnostics refreshed.' )
			
			# ----- Clear -----
			with maint_c3:
				if st.button( 'Clear Query Results', width='stretch' ):
					st.session_state[ 'semantic_result_rows' ] = [ ]
					st.session_state[ 'semantic_selected_rows' ] = [ ]
					st.session_state[ 'semantic_last_query' ] = ''
					st.success( 'Query results cleared.' )
			
			# ----- Diagnostics -----
			if bool( st.session_state.get( 'semantic_show_diagnostics', True ) ):
				st.caption(
					f'Last Query: {str( st.session_state.get( "semantic_last_query", "" ) )} '
					f'| Uploaded Sources: '
					f'{len( st.session_state.get( "semantic_uploaded_names", [ ] ) )}' )

# ==============================================================================
# PROMPT ENGINEERING MODE
# ==============================================================================
elif mode == 'Prompt Engineering':
	import math
	
	st.subheader( '📝 Prompt Engineering', help=cfg.PROMPT_ENGINEERING )
	st.divider( )
	
	# ---- Session State Initialization -----
	prompt_engineering_defaults: Dict[ str, Any ] = { 'pe_page': 1, 'pe_search': '',
		'pe_filter_category': '', 'pe_filter_category_ui': ALL_CATEGORIES_LABEL,
		'pe_sort_col': 'ID', 'pe_sort_dir': 'ASC', 'pe_selected_id': None, 'pe_caption': '',
		'pe_name': '', 'pe_edit_category': '', 'pe_text': '', 'pe_task_type': 'Chat',
		'pe_response_format': 'Markdown', 'pe_language': 'English', 'pe_generator_category': '',
		'pe_generator_goal': '', 'pe_generator_constraints': '', 'pe_generator_style': 'Practical',
		'pe_generated_template': '', 'pe_cascade_enabled': False, 'pe_jump_id': 1,
		'pe_last_search': '', 'pe_last_filter_category': '', 'pe_table_revision': 0, }
	
	for state_key, default_value in prompt_engineering_defaults.items( ):
		if state_key not in st.session_state:
			st.session_state[ state_key ] = default_value
	
	left, center, right = st.columns( [ 0.05, 0.90, 0.05 ], )
	with center:
		
		st.checkbox( label='Cascade selection into shared System Instructions and task settings',
			key='pe_cascade_enabled',
			help=('When enabled, selecting a prompt also applies its text and the current '
			      'Prompt Engineering metadata to the shared generation controls.'), )
		
		prompt_categories: List[ str ] = fetch_prompt_categories( )
		filter_category_options: List[ str ] = [ ALL_CATEGORIES_LABEL, *prompt_categories, ]
		current_filter_ui: str = str( st.session_state.get( 'pe_filter_category_ui',
			ALL_CATEGORIES_LABEL, ) or ALL_CATEGORIES_LABEL ).strip( )
		
		if current_filter_ui not in filter_category_options:
			st.session_state[ 'pe_filter_category_ui' ] = ALL_CATEGORIES_LABEL
		
		current_generator_category: str = str(
			st.session_state.get( 'pe_generator_category', '', ) or '' ).strip( )
		
		if (current_generator_category and current_generator_category not in prompt_categories):
			st.session_state[ 'pe_generator_category' ] = ''
		
		current_edit_category = str( st.session_state.get( 'pe_edit_category', '', ) or '' ).strip( )
		if (current_edit_category and current_edit_category not in prompt_categories):
			st.session_state[ 'pe_edit_category' ] = ''
		
		filter_c1, filter_c2, filter_c3, filter_c4, filter_c5 = st.columns( [ 3, 2, 2, 2, 3 ],
			gap='medium', )
		
		# ----- Search ------
		with filter_c1:
			st.text_input( label='Search', key='pe_search',
				placeholder='Caption, Name, Category, or Text',
				help='Searches Caption, Name, Category, and Text.', )
		
		# ----- Caategory ------
		with filter_c2:
			st.selectbox( label='Category', options=filter_category_options,
				key='pe_filter_category_ui', )
		
		# ----- Sort ------
		with filter_c3:
			st.selectbox( label='Sort By', options=PROMPT_SORT_COLUMNS, key='pe_sort_col', )
		
		# ----- Direction ------
		with filter_c4:
			st.selectbox( label='Direction', options=[ 'ASC', 'DESC', ], key='pe_sort_dir', )
		
		with filter_c5:
			st.markdown( """
				<div style="
					font-size: 0.95rem;
					font-weight: 600;
					margin-bottom: 0.25rem;
				">
					Go to ID
				</div>
				""", unsafe_allow_html=True, )
			
			jump_c1, jump_c2, jump_c3 = st.columns( [ 2, 1, 1 ], )
			
			# ----- Go To ------
			with jump_c1:
				st.number_input( label='Go to ID', min_value=1, step=1, key='pe_jump_id',
					label_visibility='collapsed', )
			
			# ----- Go ------
			with jump_c2:
				if st.button( label='Go', key='pe_go_to_id', width='stretch', ):
					try:
						load_prompt_record(
							prompt_id=int( st.session_state.get( 'pe_jump_id', 1, ) ) )
						
						if bool( st.session_state.get( 'pe_cascade_enabled', False, ) ):
							apply_prompt_to_shared_instructions( )
						
						st.session_state[ 'pe_table_revision' ] = int(
							st.session_state.get( 'pe_table_revision', 0 ) ) + 1
						
						st.rerun( )
					except ValueError as ex:
						st.error( str( ex ) )
			
			# ----- Clear ------
			with jump_c3:
				if st.button( label='Clear', key='pe_clear_selection_top', width='stretch', ):
					reset_prompt_selection( )
					st.rerun( )
		
		search_text: str = str( st.session_state.get( 'pe_search', '' ) or '' ).strip( )
		filter_category_ui: str = str( st.session_state.get( 'pe_filter_category_ui',
			ALL_CATEGORIES_LABEL, ) or ALL_CATEGORIES_LABEL ).strip( )
		
		filter_category: str = (
			'' if filter_category_ui == ALL_CATEGORIES_LABEL else filter_category_ui)
		
		st.session_state[ 'pe_filter_category' ] = filter_category
		last_search: str = str( st.session_state.get( 'pe_last_search', '' ) or '' )
		last_filter_category: str = str(
			st.session_state.get( 'pe_last_filter_category', '', ) or '' )
		
		if (search_text != last_search or filter_category != last_filter_category):
			st.session_state[ 'pe_page' ] = 1
			st.session_state[ 'pe_last_search' ] = search_text
			st.session_state[ 'pe_last_filter_category' ] = filter_category
		
		sort_column: str = str( st.session_state.get( 'pe_sort_col', 'ID' ) or 'ID' ).strip( )
		sort_direction: str = str( st.session_state.get( 'pe_sort_dir', 'ASC' ) or 'ASC' ).strip( )
		total_rows: int = count_prompts( search_text=search_text, category=filter_category, )
		total_pages: int = max( 1, math.ceil( total_rows / PROMPT_PAGE_SIZE ), )
		current_page: int = int( st.session_state.get( 'pe_page', 1 ) or 1 )
		current_page = max( 1, min( current_page, total_pages, ), )
		st.session_state[ 'pe_page' ] = current_page
		offset: int = (current_page - 1) * PROMPT_PAGE_SIZE
		df_prompts: pd.DataFrame = fetch_prompts_df( search_text=search_text,
			category=filter_category, sort_column=sort_column, sort_direction=sort_direction,
			limit=PROMPT_PAGE_SIZE, offset=offset, )
		
		df_prompt_table: pd.DataFrame = df_prompts.copy( )
		if 'Selected' not in df_prompt_table.columns:
			df_prompt_table.insert( 0, 'Selected', False, )
		
		selected_prompt_id: Any = st.session_state.get( 'pe_selected_id' )
		if (not df_prompt_table.empty and selected_prompt_id is not None):
			df_prompt_table[ 'Selected' ] = (
					df_prompt_table[ 'ID' ].astype( int ) == int( selected_prompt_id ))
		
		if df_prompt_table.empty:
			st.info( 'No prompt records match the current search and category filters.' )
			df_edited_prompts: pd.DataFrame = df_prompt_table.copy( )
		else:
			df_edited_prompts = st.data_editor( df_prompt_table, hide_index=True,
				use_container_width=True, disabled=[ 'ID', 'Caption', 'Name', 'Category',
					'Text', ],
				column_config={ 'Selected': st.column_config.CheckboxColumn( label='Selected',
					help='Select exactly one prompt record.', default=False, ),
					'ID': st.column_config.NumberColumn( label='ID', format='%d', ),
					'Caption': st.column_config.TextColumn( label='Caption', ),
					'Name': st.column_config.TextColumn( label='Name', ),
					'Category': st.column_config.TextColumn( label='Category', ),
					'Text': st.column_config.TextColumn( label='Text', width='large', ), },
				key=('pe_prompt_table_'
				     f'{int( st.session_state.get( "pe_table_revision", 0 ) )}'), )
		
		selected_rows: List[ Dict[ str, Any ] ] = [ ]
		if (isinstance( df_edited_prompts,
			pd.DataFrame, ) and 'Selected' in df_edited_prompts.columns):
			selected_rows = (
				df_edited_prompts.loc[ df_edited_prompts[ 'Selected' ] == True ].to_dict(
					orient='records', ))
		
		if len( selected_rows ) == 1:
			table_prompt_id: int = int( selected_rows[ 0 ][ 'ID' ] )
			if table_prompt_id != st.session_state.get( 'pe_selected_id' ):
				try:
					load_prompt_record( prompt_id=table_prompt_id, )
					
					if bool( st.session_state.get( 'pe_cascade_enabled', False, ) ):
						apply_prompt_to_shared_instructions( )
					
					st.rerun( )
				except ValueError as ex:
					st.error( str( ex ) )
		
		elif len( selected_rows ) > 1:
			st.warning( 'Select exactly one prompt row.' )
		
		page_c1, page_c2, page_c3 = st.columns( [ 0.25, 3.5, 0.25 ], )
		
		# ----- Previous ------
		with page_c1:
			if st.button( label='◀ Prev', key='pe_previous_page', width='stretch',
					disabled=current_page <= 1, ):
				st.session_state[ 'pe_page' ] = max( 1, current_page - 1, )
				st.rerun( )
		
		# ----- Page ------
		with page_c2:
			st.markdown( f'Page **{current_page}** of **{total_pages}** '
			             f'— **{total_rows:,}** matching prompt records' )
		
		# ----- Next ------
		with page_c3:
			if st.button( label='Next ▶', key='pe_next_page', width='stretch',
					disabled=current_page >= total_pages, ):
				st.session_state[ 'pe_page' ] = min( total_pages, current_page + 1, )
				st.rerun( )
		
		st.markdown( cfg.BLUE_DIVIDER, unsafe_allow_html=True, )
		
		# ----------------------------------------------------------------------------------
		# Expander - Prompt Actions
		# ----------------------------------------------------------------------------------
		with st.expander( label='⚙️ Prompt Actions', expanded=False, ):
			action_c1, action_c2, action_c3, action_c4 = st.columns( [ 0.25, 0.25, 0.25, 0.25 ], )
			
			# ----- Apply to Text ------
			with action_c1:
				if st.button( label='Apply to Text Generation', key='pe_apply_text_generation',
						width='stretch', ):
					prompt_text: str = str( st.session_state.get( 'pe_text', '' ) or '' ).strip( )
					
					if not prompt_text:
						st.error( 'Load or enter a prompt before applying it.' )
					else:
						apply_prompt_to_shared_instructions( enable_document_grounding=False, )
						
						st.success( 'Applied to shared Text Generation settings.' )
			
			# ----- Apply to Document ------
			with action_c2:
				if st.button( label='Apply to Document Q&A', key='pe_apply_document_qna',
						width='stretch', ):
					prompt_text = str( st.session_state.get( 'pe_text', '' ) or '' ).strip( )
					
					if not prompt_text:
						st.error( 'Load or enter a prompt before applying it.' )
					else:
						apply_prompt_to_shared_instructions( enable_document_grounding=True, )
						
						st.success( 'Applied to shared Document Q&A settings.' )
			
			# ----- Clone ------
			with action_c3:
				if st.button( label='Clone as New Template', key='pe_clone_prompt',
						width='stretch', ):
					try:
						clone_current_prompt( )
						st.success( 'Prompt cloned into a new editable draft.' )
						st.rerun( )
					except ValueError as ex:
						st.error( str( ex ) )
			
			# ----- Generate ------
			with action_c4:
				if st.button( label='Generate Starter Prompt', key='pe_generate_starter',
						width='stretch', ):
					generator_category: str = str(
						st.session_state.get( 'pe_generator_category', '', ) or '' ).strip( )
					
					if not generator_category:
						generator_category = (
							prompt_categories[ 0 ] if prompt_categories else 'General Chat')
					
					st.session_state[ 'pe_text' ] = build_starter_prompt_template(
						category=generator_category,
						task_type=str( st.session_state.get( 'pe_task_type', 'Chat', ) or 'Chat' ),
						response_format=str( st.session_state.get( 'pe_response_format',
							'Markdown', ) or 'Markdown' ), language=str(
							st.session_state.get( 'pe_language', 'English', ) or 'English' ), )
					
					st.session_state[ 'pe_generated_template' ] = str(
						st.session_state[ 'pe_text' ] )
					
					st.success( 'Starter prompt generated into the edit surface.' )
		
		# ----------------------------------------------------------------------------------
		# Expander - Prompt Generator
		# ----------------------------------------------------------------------------------
		with st.expander( label='🧪 Prompt Generator', expanded=False, ):
			generator_category_options: List[ str ] = (
				prompt_categories if prompt_categories else [ '' ])
			
			generator_c1, generator_c2, generator_c3, generator_c4 = st.columns(
				[ 0.25, 0.25, 0.25, 0.25 ], )
			
			# ----- Category ------
			with generator_c1:
				st.selectbox( label='Category', options=generator_category_options,
					key='pe_generator_category', disabled=len( prompt_categories ) == 0, )
			
			# ----- Task ------
			with generator_c2:
				st.selectbox( label='Task Type', options=PROMPT_TASK_TYPES, key='pe_task_type', )
			
			# ----- Format ------
			with generator_c3:
				st.selectbox( label='Response Format', options=PROMPT_RESPONSE_FORMATS,
					key='pe_response_format', )
			
			# ----- Style ------
			with generator_c4:
				st.selectbox( label='Generator Style', options=PROMPT_GENERATOR_STYLES,
					key='pe_generator_style', )
			
			st.text_input( label='Language', key='pe_language', )
			st.text_input( label='Goal', key='pe_generator_goal', )
			st.text_area( label='Constraints', height=120, key='pe_generator_constraints', )
			if st.button( label='Generate Template Draft', key='pe_generate_template_draft',
					width='stretch', ):
				generator_goal: str = str(
					st.session_state.get( 'pe_generator_goal', '', ) or '' ).strip( )
				
				if not generator_goal:
					st.error( 'Enter a prompt goal before generating a draft.' )
				else:
					draft_text: str = generate_prompt_template_draft( goal=generator_goal,
						constraints=str(
							st.session_state.get( 'pe_generator_constraints', '', ) or '' ),
						style=str( st.session_state.get( 'pe_generator_style',
							'Practical', ) or 'Practical' ),
						category=str( st.session_state.get( 'pe_generator_category', '', ) or '' ),
						task_type=str( st.session_state.get( 'pe_task_type', 'Chat', ) or 'Chat' ),
						response_format=str( st.session_state.get( 'pe_response_format',
							'Markdown', ) or 'Markdown' ), language=str(
							st.session_state.get( 'pe_language', 'English', ) or 'English' ), )
					
					st.session_state[ 'pe_generated_template' ] = draft_text
					st.session_state[ 'pe_text' ] = draft_text
					
					st.success( 'Template draft generated into the edit surface.' )
			
			generated_template: str = str(
				st.session_state.get( 'pe_generated_template', '', ) or '' )
			
			if generated_template:
				st.text_area( label='Generated Draft', value=generated_template, height=180,
					disabled=True, key='pe_generated_template_preview', )
			
			# ----- Reset ------
			if st.button( label='Reset Generator', key='pe_reset_generator', width='stretch', ):
				st.session_state[ 'pe_generator_goal' ] = ''
				st.session_state[ 'pe_generator_constraints' ] = ''
				st.session_state[ 'pe_generator_style' ] = 'Practical'
				st.session_state[ 'pe_task_type' ] = 'Chat'
				st.session_state[ 'pe_response_format' ] = 'Markdown'
				st.session_state[ 'pe_language' ] = 'English'
				st.session_state[ 'pe_generated_template' ] = ''
				
				st.session_state[ 'pe_generator_category' ] = (
					prompt_categories[ 0 ] if prompt_categories else '')
				
				st.rerun( )
		
		# ----------------------------------------------------------------------------------
		# Expander - Edit Prompt
		# ----------------------------------------------------------------------------------
		with st.expander( label='🖊️ Edit Prompt', expanded=False, ):
			edit_c1, edit_c2 = st.columns( [ 0.20, 0.80 ], gap='medium', )
			
			# ----- ID ------
			with edit_c1:
				st.text_input( label='ID', value=(
					str( st.session_state.get( 'pe_selected_id' ) ) if st.session_state.get(
						'pe_selected_id' ) is not None else ''), disabled=True,
					key='pe_selected_id_display', )
			
			# ----- Category ------
			with edit_c2:
				edit_category_options: List[ str ] = (
					prompt_categories if prompt_categories else [ '' ])
				
				st.selectbox( label='Category', options=edit_category_options,
					key='pe_edit_category', disabled=len( prompt_categories ) == 0, )
			
			st.text_input( label='Caption', key='pe_caption', max_chars=80, )
			st.text_input( label='Name', key='pe_name', max_chars=80, )
			st.text_area( label='Text', key='pe_text', height=320, max_chars=2048, )
			text_length: int = len( str( st.session_state.get( 'pe_text', '', ) or '' ) )
			st.caption( f'Text Length: {text_length:,} / 2,048 characters' )
			
			edit_action_c1, edit_action_c2, edit_action_c3 = st.columns( [ 0.34, 0.33, 0.33 ], )
			
			# ----- Save ------
			with edit_action_c1:
				save_label: str = ('💾 Save Changes' if st.session_state.get(
					'pe_selected_id' ) is not None else '➕ Create Prompt')
				
				if st.button( label=save_label, key='pe_save_prompt', width='stretch', ):
					prompt_data: Dict[ str, Any ] = {
						'Caption': st.session_state.get( 'pe_caption', '', ),
						'Name': st.session_state.get( 'pe_name', '', ),
						'Category': st.session_state.get( 'pe_edit_category', '', ),
						'Text': st.session_state.get( 'pe_text', '', ), }
					
					try:
						current_prompt_id: Any = st.session_state.get( 'pe_selected_id' )
						
						if current_prompt_id is None:
							new_prompt_id: int = insert_prompt( data=prompt_data, )
							
							load_prompt_record( prompt_id=new_prompt_id, )
							
							success_message: str = (f'Prompt ID {new_prompt_id} created.')
						else:
							update_prompt( prompt_id=int( current_prompt_id ), data=prompt_data, )
							
							load_prompt_record( prompt_id=int( current_prompt_id ), )
							
							success_message = (f'Prompt ID {int( current_prompt_id )} updated.')
						
						st.session_state[ 'pe_table_revision' ] = int(
							st.session_state.get( 'pe_table_revision', 0, ) ) + 1
						
						st.success( success_message )
						
						st.rerun( )
					except ValueError as ex:
						st.error( str( ex ) )
					except sqlite3.Error as ex:
						st.error( f'Database operation failed: {ex}' )
			
			# ----- Delete ------
			with edit_action_c2:
				if st.button( label='Delete', key='pe_delete_prompt', width='stretch',
						disabled=st.session_state.get( 'pe_selected_id' ) is None, ):
					current_prompt_id = st.session_state.get( 'pe_selected_id' )
					
					try:
						if current_prompt_id is None:
							raise ValueError( 'Select a prompt before deleting it.' )
						
						delete_prompt( prompt_id=int( current_prompt_id ) )
						
						deleted_prompt_id: int = int( current_prompt_id )
						
						reset_prompt_selection( )
						
						if st.session_state.get( 'selected_prompt_id' ) == deleted_prompt_id:
							st.session_state[ 'selected_prompt_id' ] = None
							st.session_state[ 'active_prompt_caption' ] = ''
							st.session_state[ 'active_prompt_name' ] = ''
						
						st.success( f'Prompt ID {deleted_prompt_id} deleted.' )
						
						st.rerun( )
					except ValueError as ex:
						st.error( str( ex ) )
					except sqlite3.Error as ex:
						st.error( f'Database operation failed: {ex}' )
			
			# ----- Clear ------
			with edit_action_c3:
				if st.button( label='🧹 Clear Selection', key='pe_clear_selection_bottom',
						width='stretch', ):
					reset_prompt_selection( )
					st.rerun( )

# ==============================================================================
# DATA MANAGEMENT MODE
# ==============================================================================
elif mode == 'Data Management':
	st.subheader( "🏛️ Data Management", help=cfg.DATA_MANAGEMENT )
	st.divider( )
	left, center, right = st.columns( [ 0.05, 0.90, 0.05 ] )
	with center:
		tabs = st.tabs( [ "📥 Import", "🗂 Browse", "💉 CRUD", "📊 Explore", "🔎 Filter",
		                  "🧮 Aggregate", "📈 Visualize", "⚙ Admin", "🧠 SQL" ] )
		
		tables = list_tables( )
		if not tables:
			st.info( "No tables available." )
		else:
			table = st.selectbox( "Table", tables )
			df_full = read_table( table )
		
		# ------------------------------------------------------------------------------
		# UPLOAD TAB
		# ------------------------------------------------------------------------------
		with tabs[ 0 ]:
			uploaded_file = st.file_uploader( 'Upload Excel File', type=[ 'xlsx' ] )
			overwrite = st.checkbox( 'Overwrite existing tables', value=True )
			if uploaded_file:
				try:
					sheets = pd.read_excel( uploaded_file, sheet_name=None )
					with create_connection( ) as conn:
						conn.execute( 'BEGIN' )
						for sheet_name, df in sheets.items( ):
							table_name = create_identifier( sheet_name )
							if overwrite:
								conn.execute( f'DROP TABLE IF EXISTS "{table_name}"' )
							
							# --- Create Table ---
							columns = [ ]
							df.columns = [ create_identifier( c ) for c in df.columns ]
							for col in df.columns:
								sql_type = get_sqlite_type( df[ col ].dtype )
								columns.append( f'"{col}" {sql_type}' )
							
							create_stmt = (
									f'CREATE TABLE "{table_name}" '
									f'({", ".join( columns )});'
							)
							
							conn.execute( create_stmt )
							
							# --- Insert Data ---
							placeholders = ", ".join( [ "?" ] * len( df.columns ) )
							insert_stmt = (
									f'INSERT INTO "{table_name}" '
									f'VALUES ({placeholders});'
							)
							
							conn.executemany(
								insert_stmt,
								df.where( pd.notnull( df ), None ).values.tolist( )
							)
						
						conn.commit( )
					
					st.success( 'Import completed successfully (transaction committed).' )
					st.rerun( )
				
				except Exception as e:
					try:
						conn.rollback( )
					except:
						pass
					st.error( f'Import failed — transaction rolled back.\n\n{e}' )
		
		# ------------------------------------------------------------------------------
		# BROWSE TAB
		# ------------------------------------------------------------------------------
		with tabs[ 1 ]:
			tables = list_tables( )
			if tables:
				table = st.selectbox( 'Table', tables, key='table_name' )
				df = read_table( table )
				st.dataframe( df, use_container_width=True )
			else:
				st.info( 'No tables available.' )
		
		# ------------------------------------------------------------------------------
		# CRUD (Schema-Aware)
		# ------------------------------------------------------------------------------
		with tabs[ 2 ]:
			tables = list_tables( )
			if not tables:
				st.info( 'No tables available.' )
			else:
				table = st.selectbox( 'Select Table', tables, key='crud_table' )
				df = read_table( table )
				schema = create_schema( table )
				
				# Build type map
				type_map = { col[ 1 ]: col[ 2 ].upper( ) for col in schema if col[ 1 ] != 'rowid' }
				
				# ------------------------------------------------------------------
				# INSERT
				# ------------------------------------------------------------------
				st.subheader( 'Insert Row' )
				insert_data = { }
				for column, col_type in type_map.items( ):
					if 'INT' in col_type:
						insert_data[
							column ] = st.number_input( column, step=1, key=f'ins_{column}' )
					
					elif 'REAL' in col_type:
						insert_data[
							column ] = st.number_input( column, format='%.6f', key=f'ins_{column}' )
					
					elif 'BOOL' in col_type:
						insert_data[
							column ] = 1 if st.checkbox( column, key=f'ins_{column}' ) else 0
					
					else:
						insert_data[ column ] = st.text_input( column, key=f'ins_{column}' )
				
				if st.button( 'Insert Row' ):
					cols = list( insert_data.keys( ) )
					placeholders = ', '.join( [ '?' ] * len( cols ) )
					stmt = f'INSERT INTO "{table}" ({", ".join( cols )}) VALUES ({placeholders});'
					
					with create_connection( ) as conn:
						conn.execute( stmt, list( insert_data.values( ) ) )
						conn.commit( )
					
					st.success( 'Row inserted.' )
					st.rerun( )
				
				# ------------------------------------------------------------------
				# UPDATE
				# ------------------------------------------------------------------
				st.subheader( 'Update Row' )
				rowid = st.number_input( 'Row ID', min_value=1, step=1 )
				update_data = { }
				for column, col_type in type_map.items( ):
					if 'INT' in col_type:
						val = st.number_input( column, step=1, key=f'upd_{column}' )
						update_data[ column ] = val
					
					elif 'REAL' in col_type:
						val = st.number_input( column, format='%.6f', key=f'upd_{column}' )
						update_data[ column ] = val
					
					elif 'BOOL' in col_type:
						val = 1 if st.checkbox( column, key=f'upd_{column}' ) else 0
						update_data[ column ] = val
					
					else:
						val = st.text_input( column, key=f"upd_{column}" )
						update_data[ column ] = val
				
				if st.button( 'Update Row' ):
					set_clause = ', '.join( [ f'{c}=?' for c in update_data ] )
					stmt = f'UPDATE {table} SET {set_clause} WHERE rowid=?;'
					
					with create_connection( ) as conn:
						conn.execute( stmt, list( update_data.values( ) ) + [ rowid ] )
						conn.commit( )
					
					st.success( 'Row updated.' )
					st.rerun( )
				
				# ------------------------------------------------------------------
				# DELETE
				# ------------------------------------------------------------------
				st.subheader( 'Delete Row' )
				delete_id = st.number_input( 'Row ID to Delete', min_value=1, step=1 )
				if st.button( 'Delete Row' ):
					with create_connection( ) as conn:
						conn.execute( f'DELETE FROM {table} WHERE rowid=?;', (delete_id,) )
						conn.commit( )
					
					st.success( 'Row deleted.' )
					st.rerun( )
		
		# ------------------------------------------------------------------------------
		# EXPLORE
		# ------------------------------------------------------------------------------
		with tabs[ 3 ]:
			tables = list_tables( )
			if tables:
				table = st.selectbox( 'Table', tables, key='explore_table' )
				page_size = st.slider( 'Rows per page', 10, 500, 50 )
				page = st.number_input( 'Page', min_value=1, step=1 )
				offset = (page - 1) * page_size
				df_page = read_table( table, page_size, offset )
				st.dataframe( df_page, use_container_width=True )
		
		# ------------------------------------------------------------------------------
		# FILTER
		# ------------------------------------------------------------------------------
		with tabs[ 4 ]:
			tables = list_tables( )
			if tables:
				table = st.selectbox( 'Table', tables, key='filter_table' )
				df = read_table( table )
				column = st.selectbox( 'Column', df.columns )
				value = st.text_input( 'Contains' )
				if value:
					df = df[ df[ column ].astype( str ).str.contains( value ) ]
				st.dataframe( df, use_container_width=True )
		
		# ------------------------------------------------------------------------------
		# AGGREGATE
		# ------------------------------------------------------------------------------
		with tabs[ 5 ]:
			tables = list_tables( )
			if tables:
				table = st.selectbox( 'Table', tables, key='agg_table' )
				df = read_table( table )
				numeric_cols = df.select_dtypes( include=[ 'number' ] ).columns.tolist( )
				if numeric_cols:
					col = st.selectbox( 'Column', numeric_cols )
					agg = st.selectbox( 'Function', [ 'SUM', 'AVG', 'COUNT' ] )
					if agg == 'SUM':
						st.metric( 'Result', df[ col ].sum( ) )
					elif agg == 'AVG':
						st.metric( 'Result', df[ col ].mean( ) )
					elif agg == 'COUNT':
						st.metric( 'Result', df[ col ].count( ) )
		
		# ------------------------------------------------------------------------------
		# VISUALIZE
		# ------------------------------------------------------------------------------
		with tabs[ 6 ]:
			tables = list_tables( )
			if tables:
				table = st.selectbox( 'Table', tables, key='viz_table' )
				df = read_table( table )
				numeric_cols = df.select_dtypes( include=[ 'number' ] ).columns.tolist( )
				if numeric_cols:
					col = st.selectbox( 'Column', numeric_cols, key='viz_column' )
					fig = px.histogram( df, x=col )
					st.plotly_chart( fig, use_container_width=True )
		
		# ------------------------------------------------------------------------------
		# ADMIN
		# ------------------------------------------------------------------------------
		with tabs[ 7 ]:
			tables = list_tables( )
			if tables:
				table = st.selectbox( 'Table', tables, key='admin_table' )
			
			st.divider( )
			
			st.subheader( 'Data Profiling' )
			tables = list_tables( )
			if tables:
				table = st.selectbox( 'Select Table', tables, key='profile_table' )
				if st.button( 'Generate Profile' ):
					profile_df = create_profile_table( table )
					st.dataframe( profile_df, use_container_width=True )
			
			st.subheader( 'Drop Table' )
			
			tables = list_tables( )
			if tables:
				table = st.selectbox( 'Select Table to Drop', tables, key='admin_drop_table' )
				
				# Initialize confirmation state
				if 'dm_confirm_drop' not in st.session_state:
					st.session_state.dm_confirm_drop = False
				
				# Step 1: Initial Drop click
				if st.button( 'Drop Table', key='admin_drop_button' ):
					st.session_state.dm_confirm_drop = True
				
				# Step 2: Confirmation UI
				if st.session_state.dm_confirm_drop:
					st.warning( f'You are about to permanently delete table {table}. '
					            'This action cannot be undone.' )
					
					col1, col2 = st.columns( 2 )
					
					if col1.button( 'Confirm Drop', key='admin_confirm_drop' ):
						try:
							drop_table( table )
							st.success( f'Table {table} dropped successfully.' )
						except Exception as e:
							st.error( f'Drop failed: {e}' )
						
						st.session_state.dm_confirm_drop = False
						st.rerun( )
					
					if col2.button( 'Cancel', key='admin_cancel_drop' ):
						st.session_state.dm_confirm_drop = False
						st.rerun( )
				
				df = read_table( table )
				col = st.selectbox( 'Create Index On', df.columns )
				
				if st.button( 'Create Index' ):
					create_index( table, col )
					st.success( 'Index created.' )
			
			st.divider( )
			
			st.subheader( 'Create Custom Table' )
			new_table_name = st.text_input( 'Table Name' )
			column_count = st.number_input( 'Number of Columns', min_value=1, max_value=20,
				value=1 )
			columns = [ ]
			for i in range( column_count ):
				st.markdown( f'### Column {i + 1}' )
				col_name = st.text_input( 'Column Name', key=f'col_name_{i}' )
				col_type = st.selectbox( 'Column Type', [ 'INTEGER', 'REAL', 'TEXT' ],
					key=f'col_type_{i}' )
				
				not_null = st.checkbox( 'NOT NULL', key=f'not_null_{i}' )
				primary_key = st.checkbox( 'PRIMARY KEY', key=f'pk_{i}' )
				auto_inc = st.checkbox( 'AUTOINCREMENT (INTEGER only)', key=f'ai_{i}' )
				
				columns.append( {
						'name': col_name,
						'type': col_type,
						'not_null': not_null,
						'primary_key': primary_key,
						'auto_increment': auto_inc } )
			
			if st.button( 'Create Table' ):
				try:
					create_custom_table( new_table_name, columns )
					st.success( 'Table created successfully.' )
					st.rerun( )
				
				except Exception as e:
					st.error( f'Error: {e}' )
			
			st.divider( )
			st.subheader( 'Schema Viewer' )
			
			tables = list_tables( )
			if tables:
				table = st.selectbox( 'Select Table', tables, key='schema_view_table' )
				
				# Column schema
				schema = create_schema( table )
				schema_df = pd.DataFrame(
					schema,
					columns=[ 'cid', 'name', 'type', 'notnull', 'default', 'pk' ] )
				
				st.markdown( "### Columns" )
				st.dataframe( schema_df, use_container_width=True )
				
				# Row count
				with create_connection( ) as conn:
					count = conn.execute(
						f'SELECT COUNT(*) FROM "{table}"'
					).fetchone( )[ 0 ]
				
				st.metric( "Row Count", f"{count:,}" )
				
				# Indexes
				indexes = get_indexes( table )
				if indexes:
					idx_df = pd.DataFrame(
						indexes,
						columns=[ 'seq', 'name', 'unique', 'origin', 'partial' ]
					)
					st.markdown( "### Indexes" )
					st.dataframe( idx_df, use_container_width=True )
				else:
					st.info( "No indexes defined." )
			
			st.divider( )
			st.subheader( "ALTER TABLE Operations" )
			
			tables = list_tables( )
			if tables:
				table = st.selectbox( 'Select Table', tables, key='alter_table_select' )
				operation = st.selectbox( 'Operation',
					[ 'Add Column', 'Rename Column', 'Rename Table', 'Drop Column' ] )
				
				if operation == 'Add Column':
					new_col = st.text_input( 'Column Name' )
					col_type = st.selectbox( 'Column Type', [ 'INTEGER', 'REAL', 'TEXT' ] )
					
					if st.button( 'Add Column' ):
						add_column( table, new_col, col_type )
						st.success( 'Column added.' )
						st.rerun( )
				
				elif operation == 'Rename Column':
					schema = create_schema( table )
					col_names = [ col[ 1 ] for col in schema ]
					
					old_col = st.selectbox( 'Column to Rename', col_names )
					new_col = st.text_input( 'New Column Name' )
					
					if st.button( 'Rename Column' ):
						rename_column( table, old_col, new_col )
						st.success( 'Column renamed.' )
						st.rerun( )
				
				elif operation == 'Rename Table':
					new_name = st.text_input( 'New Table Name' )
					
					if st.button( 'Rename Table' ):
						rename_table( table, new_name )
						st.success( 'Table renamed.' )
						st.rerun( )
				
				elif operation == 'Drop Column':
					schema = create_schema( table )
					col_names = [ col[ 1 ] for col in schema ]
					
					drop_col = st.selectbox( 'Column to Drop', col_names )
					
					if st.button( 'Drop Column' ):
						drop_column( table, drop_col )
						st.success( 'Column dropped.' )
						st.rerun( )
		
		# ------------------------------------------------------------------------------
		# SQL
		# ------------------------------------------------------------------------------
		with tabs[ 8 ]:
			st.subheader( 'SQL Console' )
			query = st.text_area( 'Enter SQL Query' )
			if st.button( 'Run Query' ):
				if not is_safe_query( query ):
					st.error( 'Query blocked: Only read-only SELECT statements are allowed.' )
				else:
					try:
						start_time = time.perf_counter( )
						with create_connection( ) as conn:
							result = pd.read_sql_query( query, conn )
						
						end_time = time.perf_counter( )
						elapsed = end_time - start_time
						
						# ----------------------------------------------------------
						# Display Results
						# ----------------------------------------------------------
						st.dataframe( result, use_container_width=True )
						row_count = len( result )
						
						# ----------------------------------------------------------
						# Execution Metrics
						# ----------------------------------------------------------
						col1, col2 = st.columns( 2 )
						col1.metric( 'Rows Returned', f'{row_count:,}' )
						col2.metric( 'Execution Time (seconds)', f'{elapsed:.6f}' )
						
						# Optional slow query warning
						if elapsed > 2.0:
							st.warning( 'Slow query detected (> 2 seconds). Consider indexing.' )
						
						# ----------------------------------------------------------
						# Download
						# ----------------------------------------------------------
						if not result.empty:
							csv = result.to_csv( index=False ).encode( 'utf-8' )
							st.download_button( 'Download CSV', csv,
								'query_results.csv', 'text/csv' )
					
					except Exception as e:
						st.error( f'Execution failed: {e}' )

# ==============================================================================
# FOOTER — SECTION
# ==============================================================================
st.markdown(
	"""
	<style>
	.block-container {
		padding-bottom: 3rem;
	}
	</style>
	""",
	unsafe_allow_html=True, )

# ---- Fixed Container
st.markdown(
	"""
	<style>
	.jimi-status-bar {
		position: fixed;
		bottom: 0;
		left: 0;
		width: 100%;
		background-color: rgba(27, 27, 27, 0.95);
		border-top: 1px solid #4d4d4d;
		padding: 10px 16px;
		font-size: 0.80rem;
		color: #4aa2f7;
		z-index: 1000;
	}
	.jimi-status-inner {
		display: flex;
		justify-content: space-between;
		align-items: center;
		max-width: 100%;
	}
	</style>
	""", unsafe_allow_html=True, )

# ======================================================================================
# FOOTER RENDERING
# ======================================================================================
right_parts: List[ str ] = [ ]
model = 'Jimi'
mode_value = mode if mode is not None else st.session_state.get( 'mode' )
if mode_value:
	right_parts.append( f'Mode: {mode_value}' )

temperature = st.session_state.get( 'temperature' )
top_p = st.session_state.get( 'top_percent' )
top_k = st.session_state.get( 'top_k' )
frequency = st.session_state.get( 'frequency_penalty' )
presence = st.session_state.get( 'presence_penalty' )
repeat_penalty = st.session_state.get( 'repeat_penalty' )
max_tokens = st.session_state.get( 'max_tokens' )
context_window = st.session_state.get( 'context_window' )
cpu_threads = st.session_state.get( 'cpu_threads' )
repeat_window = st.session_state.get( 'repeat_window' )
use_semantic = st.session_state.get( 'use_semantic' )
basic_docs = st.session_state.get( 'basic_docs' )

# ------------------------------------------------------------------
# Parameter summary (show 0 values; suppress only when None)
# ------------------------------------------------------------------
if temperature is not None:
	right_parts.append( f'Temp: {float( temperature ):0.2f}' )

if top_p is not None:
	right_parts.append( f'Top-P: {float( top_p ):0.2f}' )

if top_k is not None:
	right_parts.append( f'Top-K: {int( top_k )}' )

if frequency is not None:
	right_parts.append( f'Freq: {float( frequency ):0.2f}' )

if presence is not None:
	right_parts.append( f'Presence: {float( presence ):0.2f}' )

if repeat_penalty is not None:
	right_parts.append( f'Repeat: {float( repeat_penalty ):0.2f}' )

if repeat_window is not None:
	right_parts.append( f'Repeat Window: {int( repeat_window )}' )

if max_tokens is not None:
	right_parts.append( f'Max Tokens: {int( max_tokens )}' )

if context_window is not None:
	right_parts.append( f'Context: {int( context_window )}' )

if cpu_threads is not None:
	right_parts.append( f'Threads: {int( cpu_threads )}' )

# ------------------------------------------------------------------
# Context flags (optional but useful)
# ------------------------------------------------------------------
if use_semantic is not None:
	right_parts.append( f'Semantic: {"On" if use_semantic else "Off"}' )

if isinstance( basic_docs, list ):
	right_parts.append( f'Docs: {len( basic_docs )}' )

right_text = ' ◽ '.join( right_parts ) if right_parts else '—'

# ---- Rendering Method
st.markdown(
	f"""
    <div class="jimi-status-bar">
        <div class="jimi-status-inner">
            <span>{model}</span>
            <span>{right_text}</span>
        </div>
    </div>
    """, unsafe_allow_html=True, )