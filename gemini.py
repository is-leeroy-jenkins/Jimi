'''
  ******************************************************************************************
      Assembly:                Jeni
      Filename:                gemini.py
      Author:                  Terry D. Eppler
      Created:                 05-31-2022

      Last Modified By:        Terry D. Eppler
      Last Modified On:        12-27-2025
  ******************************************************************************************
  <copyright file="gemini.py" company="Terry D. Eppler">

	     gemini.py
	     Copyright ©  2024  Terry Eppler

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
    gemini.py
  </summary>
  ******************************************************************************************
'''
from google.genai.file_search_stores import FileSearchStores
import config as cfg
import base64
from boogr import Error, Logger
import json
import os
import requests
import PIL.Image
from pathlib import Path
from typing import Any, List, Optional, Dict, Union
from google import genai
from google.cloud import storage
from google.genai import types
from google.genai.pagers import Pager
from google.genai.types import (Part, GenerateContentConfig, ImageConfig, FunctionCallingConfig,
                                GenerateImagesConfig, GenerateVideosConfig, ThinkingConfig,
                                GeneratedImage, EmbedContentConfig, Content, ContentEmbedding,
                                Candidate, HttpOptions, GenerateImagesResponse, Field,
                                FileSearchStore,
                                GenerateContentResponse, GenerateVideosResponse, Image, File,
                                SpeakerVoiceConfig, VoiceConfig, SpeechConfig, Tool, ToolConfig,
                                GoogleSearch, UrlContext, SafetySetting, HarmCategory,
                                HarmBlockThreshold)

def throw_if( name: str, value: object ) -> None:
	"""Validate a required runtime value.

	Purpose:
		Validates a required runtime value before provider request construction. The helper raises
		ValueError when a required prompt, path, model, identifier, or configuration value is
		missing so callers fail before sending incomplete requests to Gemini.

	Args:
		name: Name value used by the active workflow, such as an argument name, display name,
		or remote object name.
		value: Runtime value to validate or normalize.
	"""
	if value is None:
		raise ValueError( f'Argument "{name}" cannot be empty!' )

def encode_image( image_path: str ) -> str:
	"""Encode a local image as base64 text.

	Purpose:
		Reads a local image file and returns its base64 representation for Gemini vision or
		multimodal request payloads that require encoded image content.

	Args:
		image_path: Local path to the image file that will be read and encoded.

	Returns:
		str: Text output produced by the workflow.
	"""
	with open( image_path, "rb" ) as image_file:
		return base64.b64encode( image_file.read( ) ).decode( 'utf-8' )

class Gemini( ):
	"""Define Gemini provider workflow state.

	Purpose:
		Stores the shared Gemini provider configuration used by all wrapper classes. The class
		centralizes API keys, model identifiers, generation controls, response holders,
		tool settings, and modality state so specialized workflows can build provider requests
		from a common runtime contract.

	Attributes:
		number: Candidate count or output count requested from the provider.
		google_api_key: Google API key read from project configuration.
		gemini_api_key: Gemini API key read from project configuration.
		instructions: System instructions supplied to generation configuration.
		prompt: Prompt, instruction, or query used by the active workflow.
		model: Gemini model identifier used by provider requests.
		api_version: API version or HTTP option value used for provider clients.
		max_tokens: Maximum output token count used by generation workflows.
		temperature: Sampling temperature used by generation workflows.
		top_p: Nucleus sampling threshold used by generation workflows.
		top_k: Top-k sampling threshold used by generation workflows.
		candidate_count: Number of response candidates requested from the provider.
		media_resolution: Media-resolution setting used by multimodal requests.
		response_modalities: Output modalities requested from Gemini.
		stops: Stop sequences supplied to provider configuration.
		domains: Domain filters or grounding-domain state used by request workflows.
		frequency_penalty: Frequency penalty supplied to generation configuration.
		presence_penalty: Presence penalty supplied to generation configuration.
		response_format: Response MIME type or output format requested from the provider.
		content_response: Most recent Gemini content-generation response.
		image_response: Most recent Gemini image-generation response.
		content_config: GenerateContentConfig used by the active request.
		function_config: Function-calling configuration used by tool workflows.
		thought_config: Thinking or reasoning configuration used by generation workflows.
		genimg_config: Image-generation configuration used by image workflows.
		image_config: Image-specific request configuration.
		tool_config: Tool configuration supplied to Gemini requests.
		tool_choice: Tool-selection mode supplied to provider configuration.
		tools: Tool names selected for the active request.
	"""
	number: Optional[ int ]
	google_api_key: Optional[ str ]
	gemini_api_key: Optional[ str ]
	instructions: Optional[ str ]
	prompt: Optional[ str ]
	model: Optional[ str ]
	api_version: Optional[ str ]
	max_tokens: Optional[ int ]
	temperature: Optional[ float ]
	top_p: Optional[ float ]
	top_k: Optional[ int ]
	candidate_count: Optional[ int ]
	media_resolution: Optional[ str ]
	response_modalities: Optional[ List[ str ] ]
	stops: Optional[ List[ str ] ]
	domains: Optional[ List[ str ] ]
	frequency_penalty: Optional[ float ]
	presence_penalty: Optional[ float ]
	response_format: Optional[ str ]
	content_response: Optional[ GenerateContentResponse ]
	image_response: Optional[ GenerateImagesResponse ]
	content_config: Optional[ GenerateContentConfig ]
	function_config: Optional[ FunctionCallingConfig ]
	thought_config: Optional[ ThinkingConfig ]
	genimg_config: Optional[ GenerateImagesConfig ]
	image_config: Optional[ ImageConfig ]
	tool_config: Optional[ List[ types.Tool ] ]
	tool_choice: Optional[ str ]
	tools: Optional[ List[ str ] ]
	
	def __init__( self ) -> None:
		"""Initialize Gemini runtime state.

		Purpose:
			Initializes Gemini runtime attributes without executing provider requests. The
			constructor prepares configuration fields, client placeholders, request state,
			response holders, and workflow-specific caches used by later methods.
		"""
		self.google_api_key = cfg.GOOGLE_API_KEY
		self.gemini_api_key = cfg.GEMINI_API_KEY
		self.model = None
		self.api_version = None
		self.temperature = None
		self.top_p = None
		self.top_k = None
		self.candidate_count = None
		self.frequency_penalty = None
		self.presence_penalty = None
		self.max_tokens = None
		self.instructions = None
		self.prompt = None
		self.response_format = None
		self.number = None
		self.response_modalities = [ ]
		self.stops = [ ]
		self.tools = [ ]

class Chat( Gemini ):
	"""Define Chat provider workflow state.

	Purpose:
		Coordinates Gemini text generation, tool configuration, grounding, structured-output,
		and conversation-history workflows. The class validates request inputs, builds Gemini
		content objects, prepares GenerateContentConfig settings, creates provider clients at call
		time, and returns text output for Jimi chat operations.

	Attributes:
		use_vertex: Indicates whether Vertex AI behavior is enabled for compatible workflows.
		http_options: HTTP and API-version settings used by provider clients.
		client: Provider client used to execute Gemini requests.
		storage_client: Google Cloud Storage client used by storage-backed workflows.
		contents: Content payload sent to Gemini.
		image_uri: Image URI used by multimodal workflows.
		audio_uri: Audio URI used by audio workflows.
		file_path: Local file path used by file, image, audio, or document workflows.
		files: Collection of file names or file objects tracked by the wrapper.
		content_block: Optional content block prepended to the active prompt.
		context: Prior conversation context used to build Gemini Content objects.
		urls: URLs appended to request content for grounded workflows.
		max_urls: Maximum number of URLs retained for request content.
		response_schema: Structured-output schema supplied to generation configuration.
		safety_profile: Named safety profile selected for request configuration.
		safety_settings: Safety settings supplied to Gemini requests.
	"""
	use_vertex: Optional[ bool ]
	http_options: Optional[ HttpOptions ]
	client: Optional[ genai.Client ]
	storage_client: Optional[ storage.Client ]
	contents: Optional[ Union[ str, List[ str ], List[ Content ] ] ]
	image_uri: Optional[ str ]
	audio_uri: Optional[ str ]
	file_path: Optional[ str ]
	files: Optional[ List[ str ] ]
	content_block: Optional[ str ]
	context: Optional[ List[ Dict[ str, Any ] ] ]
	urls: Optional[ List[ str ] ]
	max_urls: Optional[ int ]
	response_schema: Optional[ Any ]
	safety_profile: Optional[ str ]
	safety_settings: Optional[ List[ SafetySetting ] ]
	
	def __init__( self, model: str = 'gemini-2.5-flash-lite' ) -> None:
		"""Initialize Chat runtime state.

		Purpose:
			Initializes Chat runtime attributes without executing provider requests. The
			constructor prepares configuration fields, client placeholders, request state,
			response holders, and workflow-specific caches used by later methods.

		Args:
			model: Gemini model identifier used for the provider request.
		"""
		super( ).__init__( )
		self.api_version = None
		self.client = None
		self.content_config = None
		self.image_config = None
		self.function_tool_config = None
		self.thought_config = None
		self.genimg_config = None
		self.tool_objects = None
		self.tools = [ ]
		self.response_modalities = [ ]
		self.files = [ ]
		self.http_options = { }
		self.number = None
		self.candidate_count = None
		self.model = model
		self.top_p = None
		self.top_k = None
		self.temperature = None
		self.frequency_penalty = None
		self.presence_penalty = None
		self.max_tokens = None
		self.use_vertex = None
		self.instructions = None
		self.media_resolution = None
		self.tool_choice = None
		self.contents = None
		self.content_block = None
		self.context = [ ]
		self.client = None
		self.storage_client = None
		self.content_response = None
		self.image_response = None
		self.image_uri = None
		self.audio_uri = None
		self.file_path = None
		self.stops = [ ]
		self.response_mime_type = None
		self.response_schema = None
		self.urls = [ ]
		self.max_urls = None
		self.safety_profile = None
		self.safety_settings = None
	
	@property
	def model_options( self ) -> List[ str ] | None:
		"""Model options.

		Purpose:
			Returns the model identifiers exposed by this wrapper for user-interface selectors and
			request configuration.

		Returns:
			List[ str ] | None: List of string values when available; otherwise None.
		"""
		return [ 'gemini-2.5-flash',
		         'gemini-2.5-flash-lite',
		         'gemini-2.5-pro',
		         'gemini-3-flash-preview',
		         'gemini-3.1-flash-lite-preview',
		         'gemini-3.1-pro-preview',
		         'gemini-2.0-flash',
		         'gemini-2.0-flash-lite' ]
	
	@property
	def tool_options( self ) -> List[ str ] | None:
		"""Tool options.

		Purpose:
			Returns the tool names exposed by this wrapper for user-interface selectors and
			provider request configuration.

		Returns:
			List[ str ] | None: List of string values when available; otherwise None.
		"""
		return [ 'google_search',
		         'google_maps',
		         'url_context',
		         'code_execution' ]
	
	@property
	def reasoning_options( self ) -> List[ str ] | None:
		"""Reasoning options.

		Purpose:
			Returns the reasoning-level names exposed by this wrapper for user-interface selectors
			and thinking configuration.

		Returns:
			List[ str ] | None: List of string values when available; otherwise None.
		"""
		return [ 'THINKING_LEVEL_UNSPECIFIED', 'MINIMAL',
		         'LOW', 'MEDIUM', 'HIGH' ]
	
	@property
	def media_options( self ) -> List[ str ] | None:
		"""Media options.

		Purpose:
			Returns media-resolution option names exposed by this wrapper for multimodal request
			configuration.

		Returns:
			List[ str ] | None: List of string values when available; otherwise None.
		"""
		return [ 'media_resolution_high',
		         'media_resolution_medium',
		         'media_resolution_low' ]
	
	@property
	def choice_options( self ) -> List[ str ] | None:
		"""Choice options.

		Purpose:
			Returns tool-choice option names exposed by this wrapper for provider tool-selection
			configuration.

		Returns:
			List[ str ] | None: List of string values when available; otherwise None.
		"""
		return [ 'AUTO',
		         'ANY',
		         'NONE',
		         'VALIDATED' ]
	
	@property
	def include_options( self ) -> List[ str ] | None:
		"""Include options.

		Purpose:
			Returns include-field option names exposed by this wrapper for response expansion and
			diagnostic request configuration.

		Returns:
			List[ str ] | None: List of string values when available; otherwise None.
		"""
		return [ 'file_search_call.results',
		         'message.input_image.image_url',
		         'message.output_text.logprobs',
		         'reasoning.encrypted_content' ]
	
	@property
	def modality_options( self ) -> List[ str ] | None:
		"""Modality options.

		Purpose:
			Returns response-modality option names exposed by this wrapper for text, image,
			and audio request configuration.

		Returns:
			List[ str ] | None: List of string values when available; otherwise None.
		"""
		return [ 'MODALITY_UNSPECIFIED', 'TEXT', 'IMAGE', 'AUDIO' ]
	
	@property
	def format_options( self ) -> List[ str ] | None:
		"""Format options.

		Purpose:
			Returns response-format option names exposed by this wrapper for MIME-type or
			content-format selection.

		Returns:
			List[ str ] | None: List of string values when available; otherwise None.
		"""
		return [ 'text/plain',
		         'application/json',
		         'text/x.enum' ]
	
	@property
	def safety_options( self ) -> List[ str ] | None:
		"""Safety options.

		Purpose:
			Returns safety profile names exposed by the chat wrapper for safety-setting
			construction.

		Returns:
			List[ str ] | None: List of string values when available; otherwise None.
		"""
		return [ '',
		         'strict',
		         'balanced',
		         'permissive' ]
	
	def get_supported_tool_options( self, model: str = None ) -> List[ str ]:
		"""Get supported tool options.

		Purpose:
			Computes the built-in Gemini tool names supported by the selected chat model. The
			method always exposes core grounding and execution tools and conditionally adds Google
			Maps when the active model supports it.

		Args:
			model: Gemini model identifier used for the provider request.

		Returns:
			List[ str ]: List of string values produced by the workflow.

		Raises:
			Error: Raised when validation, provider execution, storage access, or response
		processing fails.
		"""
		try:
			self.model_name = str( model or self.model or '' ).strip( ).lower( )
			self.options = [ 'google_search', 'url_context', 'code_execution' ]
			
			if self._supports_google_maps( self.model_name ):
				self.options.append( 'google_maps' )
			
			return self.options
		except Exception as e:
			exception = Error( e )
			exception.module = 'gemini'
			exception.cause = 'Chat'
			exception.method = 'get_supported_tool_options( self, model: str=None )'
			Logger( ).write( exception )
			raise exception
	
	def _resolve_api_key( self ) -> str | None:
		"""Resolve api key.

		Purpose:
			Resolves the Gemini API key at call time from environment variables and project
			configuration. The method supports flexible deployment while keeping provider-client
			creation centralized inside request methods.

		Returns:
			str | None: Text output when available; otherwise None.

		Raises:
			Error: Raised when validation, provider execution, storage access, or response
		processing fails.
		"""
		try:
			self.api_key = os.environ.get( 'GEMINI_API_KEY' )
			if self.api_key and str( self.api_key ).strip( ):
				return str( self.api_key ).strip( )
			
			self.api_key = os.environ.get( 'GOOGLE_API_KEY' )
			if self.api_key and str( self.api_key ).strip( ):
				return str( self.api_key ).strip( )
			
			self.api_key = getattr( cfg, 'GEMINI_API_KEY', None )
			if self.api_key and str( self.api_key ).strip( ):
				return str( self.api_key ).strip( )
			
			self.api_key = getattr( cfg, 'GOOGLE_API_KEY', None )
			if self.api_key and str( self.api_key ).strip( ):
				return str( self.api_key ).strip( )
			
			return None
		except Exception as e:
			exception = Error( e )
			exception.module = 'gemini'
			exception.cause = 'Chat'
			exception.method = '_resolve_api_key( self ) -> str | None'
			Logger( ).write( exception )
			raise exception
	
	def _supports_google_maps( self, model: str = None ) -> bool:
		"""Supports google maps.

		Purpose:
			Determines whether the selected Gemini model should expose Google Maps grounding in
			the Jimi user interface and request builder.

		Args:
			model: Gemini model identifier used for the provider request.

		Returns:
			bool: Boolean result produced by the workflow.

		Raises:
			Error: Raised when validation, provider execution, storage access, or response
processing fails.
		"""
		try:
			self.model_name = str( model or self.model or '' ).strip( ).lower( )
			self.maps_models = {
					'gemini-3.1-pro-preview',
					'gemini-3.1-flash-lite-preview',
					'gemini-3-flash-preview',
					'gemini-2.5-pro',
					'gemini-2.5-flash',
					'gemini-2.5-flash-lite',
					'gemini-2.0-flash'
			}
			return self.model_name in self.maps_models
		except Exception as e:
			exception = Error( e )
			exception.module = 'gemini'
			exception.cause = 'Chat'
			exception.method = '_supports_google_maps( self, model: str=None ) -> bool'
			Logger( ).write( exception )
			raise exception
	
	def _supports_computer_use( self, model: str = None ) -> bool:
		"""Supports computer use.

		Purpose:
			Determines whether the selected Gemini model should expose computer-use tooling. The
			current wrapper returns False to keep unsupported tools out of request configuration.

		Args:
			model: Gemini model identifier used for the provider request.

		Returns:
			bool: Boolean result produced by the workflow.

		Raises:
			Error: Raised when validation, provider execution, storage access, or response
			processing fails.
		"""
		try:
			return False
		except Exception as e:
			exception = Error( e )
			exception.module = 'gemini'
			exception.cause = 'Chat'
			exception.method = '_supports_computer_use( self, model: str=None ) -> bool'
			Logger( ).write( exception )
			raise exception
	
	def _normalize_positive_int( self, value: Any = None ) -> int | None:
		"""Normalize positive int.

		Purpose:
			Normalizes optional integer request parameters. Positive values are returned for
			provider configuration, while None, zero, and invalid values are omitted from the
			request.

		Args:
			value: Runtime value to validate or normalize.

		Returns:
			int | None: int | None value produced by the workflow.
		"""
		try:
			if value is None:
				return None
			
			self.int_value = int( value )
			return self.int_value if self.int_value > 0 else None
		except Exception as e:
			ex = Error( e )
			ex.module = 'gemini'
			ex.cause = 'Chat'
			ex.method = '_normalize_positive_int( self, *args )'
			Logger( ).write( ex )
			return None
	
	def _build_tools( self, tools: List[ str ] = None ) -> List[ Tool ] | None:
		"""Build tools.

		Purpose:
			Builds Gemini Tool objects from user-selected tool names. The method filters
			unsupported names, maps supported options to SDK tool instances, and returns the
			configured collection for GenerateContentConfig.

		Args:
			tools: Tool names selected for provider request configuration.

		Returns:
			List[ Tool ] | None: List[ Tool ] | None value produced by the workflow.

		Raises:
			Error: Raised when validation, provider execution, storage access, or response
			processing fails.
		"""
		try:
			self.tools = tools if tools is not None else [ ]
			self.tool_objects = [ ]
			self.supported_tools = set( self.get_supported_tool_options( self.model ) )
			
			for name in self.tools:
				if not isinstance( name, str ):
					continue
				
				self.name = str( name ).strip( )
				if not self.name or self.name not in self.supported_tools:
					continue
				
				if self.name == 'google_search':
					self.tool_objects.append( Tool( google_search=GoogleSearch( ) ) )
				
				elif self.name == 'url_context':
					self.tool_objects.append( Tool( url_context=UrlContext( ) ) )
				
				elif self.name == 'code_execution':
					self.tool_objects.append( Tool( code_execution=types.ToolCodeExecution ) )
				
				elif self.name == 'google_maps':
					self.tool_objects.append( Tool( google_maps=types.GoogleMaps( ) ) )
			
			return self.tool_objects if len( self.tool_objects ) > 0 else None
		except Exception as e:
			exception = Error( e )
			exception.module = 'gemini'
			exception.cause = 'Chat'
			exception.method = '_build_tools( self, tools: List[ str ]=None )'
			Logger( ).write( exception )
			raise exception
	
	def _parse_response_schema( self, response_schema: Any = None ) -> Any:
		"""Parse response schema.

		Purpose:
			Normalizes structured-output schema input supplied as a dictionary, JSON string,
			or schema object. The method returns a provider-ready schema or None when no usable
			schema is supplied.

		Args:
			response_schema: Structured-output schema supplied as a dictionary, JSON string,
			or schema object.

		Returns:
			Any: Provider-specific object produced by the workflow.

		Raises:
			Error: Raised when validation, provider execution, storage access, or response
			processing fails.
		"""
		try:
			if response_schema is None:
				return None
			
			if isinstance( response_schema, dict ):
				return response_schema if len( response_schema ) > 0 else None
			
			if isinstance( response_schema, str ):
				self.schema_text = response_schema.strip( )
				if not self.schema_text:
					return None
				
				return json.loads( self.schema_text )
			
			return response_schema
		except Exception as e:
			exception = Error( e )
			exception.module = 'gemini'
			exception.cause = 'Chat'
			exception.method = '_parse_response_schema( self, response_schema: Any=None )'
			Logger( ).write( exception )
			raise exception
	
	def _build_contents( self, prompt: str, context: List[ Any ] = None,
			content: str = None ) -> str | List[ Content ]:
		"""Build contents.

		Purpose:
			Builds Gemini Content objects from the active prompt, optional prepended content,
			and prior conversation context. The method converts assistant messages to model-role
			content and user messages to user-role content for provider requests.

		Args:
			prompt: User prompt, instruction, or query submitted to the provider workflow.
			context: Prior conversation context supplied as dictionaries or Gemini Content objects.
			content: Optional text block prepended to the active prompt.

		Returns:
			str | List[ Content ]: str | List[ Content ] value produced by the workflow.

		Raises:
			Error: Raised when validation, provider execution, storage access, or response
 processing fails.
		"""
		try:
			throw_if( 'prompt', prompt )
			self.prompt = prompt
			self.context = context if context is not None else [ ]
			self.content_block = content
			self.contents = [ ]
			for item in self.context:
				if item is None:
					continue
				
				if isinstance( item, Content ):
					self.contents.append( item )
					continue
				
				if not isinstance( item, dict ):
					continue
				
				role = item.get( 'role', 'user' )
				text = item.get( 'content', None )
				if text is None:
					continue
				
				text = str( text ).strip( )
				if not text:
					continue
				
				if role == 'assistant':
					self.contents.append(
						Content(
							role='model',
							parts=[ Part.from_text( text=text ) ]
						)
					)
				else:
					self.contents.append(
						Content(
							role='user',
							parts=[ Part.from_text( text=text ) ]
						)
					)
			
			self.user_text = str( self.prompt ).strip( )
			if self.content_block is not None and str( self.content_block ).strip( ):
				self.user_text = f"{str( self.content_block ).strip( )}\n\n{self.user_text}"
			
			self.contents.append(
				Content(
					role='user',
					parts=[ Part.from_text( text=self.user_text ) ]
				)
			)
			
			return self.contents if len( self.contents ) > 0 else self.prompt
		except Exception as e:
			exception = Error( e )
			exception.module = 'gemini'
			exception.cause = 'Chat'
			exception.method = ('_build_contents( self, prompt: str, '
			                    'context: List[ Any ]=None, '
			                    'content: str=None )')
			Logger( ).write( exception )
			raise exception
	
	def _get_response_content( self ) -> Content | None:
		"""Get response content.

		Purpose:
			Extracts the model Content block from the most recent Gemini content response. The
			method supports structured conversation history by returning the first candidate
			content object when available.

		Returns:
			Content | None: Gemini content object when available; otherwise None.

		Raises:
			Error: Raised when validation, provider execution, storage access, or response
			processing fails.
		"""
		try:
			if self.content_response is None:
				return None
			
			if hasattr( self.content_response, 'candidates' ) and self.content_response.candidates:
				for candidate in self.content_response.candidates:
					if hasattr( candidate, 'content' ) and candidate.content is not None:
						return candidate.content
			
			return None
		except Exception as e:
			exception = Error( e )
			exception.module = 'gemini'
			exception.cause = 'Chat'
			exception.method = '_get_response_content( self ) -> Content | None'
			Logger( ).write( exception )
			raise exception
	
	def get_structured_history( self ) -> List[ Content ] | None:
		"""Get structured history.

		Purpose:
			Builds structured Gemini conversation history from the last request contents and
			response content. The returned history can be reused in a later request to preserve
			chat context.

		Returns:
			List[ Content ] | None: List[ Content ] | None value produced by the workflow.

		Raises:
			Error: Raised when validation, provider execution, storage access, or response
 processing fails.
		"""
		try:
			self.history = [ ]
			
			if self.contents is not None and isinstance( self.contents, list ):
				for item in self.contents:
					if isinstance( item, Content ):
						self.history.append( item )
			
			self.response_content = self._get_response_content( )
			if self.response_content is not None:
				self.history.append( self.response_content )
			
			return self.history if len( self.history ) > 0 else None
		except Exception as e:
			exception = Error( e )
			exception.module = 'gemini'
			exception.cause = 'Chat'
			exception.method = 'get_structured_history( self ) -> List[ Content ] | None'
			Logger( ).write( exception )
			raise exception
	
	def _build_config( self, model: str = 'gemini-2.5-flash-lite', number: int = None,
			temperature: float = None, top_p: float = None, top_k: int = None,
			frequency: float = None, presence: float = None, max_tokens: int = None,
			stops: List[ str ] = None, instruct: str = None, response_format: str = None,
			tools: List[ str ] = None, tool_choice: str = None, reasoning: str = None,
			modalities: List[ str ] = None, media_resolution: str = None,
			response_schema: Any = None, safety_profile: str = None ) -> GenerateContentConfig:
		"""Build config.

		Purpose:
			Builds the GenerateContentConfig object for Gemini chat generation. The method
			normalizes generation controls, safety settings, tools, modalities, reasoning
			configuration, response format, and structured-output schema before provider execution.

		Args:
			model: Gemini model identifier used for the provider request.
			number: Candidate count or output count requested from the provider.
			temperature: Sampling temperature applied to generation when supplied.
			top_p: Nucleus sampling threshold applied to generation when supplied.
			top_k: Top-k sampling threshold applied to generation when supplied.
			frequency: Frequency penalty applied to repeated tokens when supplied.
			presence: Presence penalty applied to repeated topics or tokens when supplied.
			max_tokens: Maximum output token count supplied to provider configuration.
			stops: Stop sequences supplied to generation configuration.
			instruct: Optional system instruction supplied to provider configuration.
			response_format: Response MIME type or structured-output format requested from the
			provider.
			tools: Tool names selected for provider request configuration.
			tool_choice: Provider tool-selection mode.
			reasoning: Reasoning or thinking-level option supplied to provider configuration.
			modalities: Response modalities requested from the provider.
			media_resolution: Media-resolution option supplied for multimodal requests.
			response_schema: Structured-output schema supplied as a dictionary, JSON string,
			or schema object.
			safety_profile: Named safety profile used to build Gemini safety settings.

		Returns:
			GenerateContentConfig: Provider-ready content generation configuration.

		Raises:
			Error: Raised when validation, provider execution, storage access, or response
 processing fails.
		"""
		try:
			self.model = str( model or self.model or 'gemini-2.5-flash-lite' ).strip( )
			self.number = self._normalize_positive_int( number )
			self.candidate_count = self.number
			self.temperature = temperature
			self.top_p = top_p
			self.top_k = self._normalize_positive_int( top_k )
			self.frequency_penalty = frequency
			self.presence_penalty = presence
			self.max_tokens = self._normalize_positive_int( max_tokens )
			self.stops = stops if stops is not None else [ ]
			self.instructions = instruct
			self.response_mime_type = response_format
			self.response_schema = self._parse_response_schema( response_schema=response_schema )
			self.safety_settings = self._build_safety_settings( safety_profile=safety_profile )
			self.tool_choice = tool_choice
			self.media_resolution = str( media_resolution ).strip( ) if media_resolution else None
			self.tool_objects = self._build_tools( tools=tools )
			self.function_tool_config = self._build_tool_config(
				tool_choice=self.tool_choice,
				tools=self.tool_objects )
			self.response_modalities = self._build_modalities( modalities=modalities )
			self.thought_config = self._build_reasoning( reasoning=reasoning )
			self.config_kwargs = { }
			
			if self.temperature is not None:
				self.config_kwargs[ 'temperature' ] = self.temperature
			
			if self.top_p is not None:
				self.config_kwargs[ 'top_p' ] = self.top_p
			
			if self.top_k is not None:
				self.config_kwargs[ 'top_k' ] = self.top_k
			
			if self.max_tokens is not None:
				self.config_kwargs[ 'max_output_tokens' ] = self.max_tokens
			
			if self.candidate_count is not None:
				self.config_kwargs[ 'candidate_count' ] = self.candidate_count
			
			if self.instructions is not None and str( self.instructions ).strip( ):
				self.config_kwargs[ 'system_instruction' ] = str( self.instructions ).strip( )
			
			if self.frequency_penalty is not None:
				self.config_kwargs[ 'frequency_penalty' ] = self.frequency_penalty
			
			if self.presence_penalty is not None:
				self.config_kwargs[ 'presence_penalty' ] = self.presence_penalty
			
			if self.stops is not None and len( self.stops ) > 0:
				self.config_kwargs[ 'stop_sequences' ] = self.stops
			
			if self.response_mime_type is not None and str( self.response_mime_type ).strip( ):
				self.config_kwargs[ 'response_mime_type' ] = str( self.response_mime_type
				).strip( )
			
			if self.response_schema is not None:
				if isinstance( self.response_schema, dict ):
					self.config_kwargs[ 'response_json_schema' ] = self.response_schema
				else:
					self.config_kwargs[ 'response_schema' ] = self.response_schema
			
			if self.media_resolution is not None and self.media_resolution:
				self.config_kwargs[ 'media_resolution' ] = self.media_resolution
			
			if self.tool_objects is not None and len( self.tool_objects ) > 0:
				self.config_kwargs[ 'tools' ] = self.tool_objects
			
			if self.function_tool_config is not None:
				self.config_kwargs[ 'tool_config' ] = self.function_tool_config
			
			if self.safety_settings is not None and len( self.safety_settings ) > 0:
				self.config_kwargs[ 'safety_settings' ] = self.safety_settings
			
			if self.response_modalities is not None and len( self.response_modalities ) > 0:
				self.config_kwargs[ 'response_modalities' ] = self.response_modalities
			
			if self.thought_config is not None:
				self.config_kwargs[ 'thinking_config' ] = self.thought_config
			
			self.content_config = GenerateContentConfig( **self.config_kwargs )
			return self.content_config
		except Exception as e:
			exception = Error( e )
			exception.module = 'gemini'
			exception.cause = 'Chat'
			exception.method = '_build_config( self, model ) -> GenerateContentConfig'
			Logger( ).write( exception )
			raise exception
	
	def generate_text( self, prompt: str, model: str = 'gemini-2.5-flash-lite',
			number: int = None, temperature: float = None, top_p: float = None, top_k: int = None,
			frequency: float = None, presence: float = None, max_tokens: int = None,
			stops: List[ str ] = None, instruct: str = None, response_format: str = None,
			tools: List[ str ] = None, tool_choice: str = None, reasoning: str = None,
			modalities: List[ str ] = None, media_resolution: str = None,
			context: List[ Dict[ str, Any ] ] = None, content: str = None,
			urls: List[ str ] = None, max_urls: int = None, response_schema: Any = None,
			safety_profile: str = None, stream: bool = False,
			stream_handler: Any = None ) -> str | None:
		"""Generate text with Gemini.

		Purpose:
			Runs a Gemini text-generation request using the supplied prompt and generation
			settings. The method validates input, builds content and configuration objects,
			creates the GenAI client, supports optional streaming, and returns extracted text
			output.

		Args:
			prompt: User prompt, instruction, or query submitted to the provider workflow.
			model: Gemini model identifier used for the provider request.
			number: Candidate count or output count requested from the provider.
			temperature: Sampling temperature applied to generation when supplied.
			top_p: Nucleus sampling threshold applied to generation when supplied.
			top_k: Top-k sampling threshold applied to generation when supplied.
			frequency: Frequency penalty applied to repeated tokens when supplied.
			presence: Presence penalty applied to repeated topics or tokens when supplied.
			max_tokens: Maximum output token count supplied to provider configuration.
			stops: Stop sequences supplied to generation configuration.
			instruct: Optional system instruction supplied to provider configuration.
			response_format: Response MIME type or structured-output format requested from the
			provider.
			tools: Tool names selected for provider request configuration.
			tool_choice: Provider tool-selection mode.
			reasoning: Reasoning or thinking-level option supplied to provider configuration.
			modalities: Response modalities requested from the provider.
			media_resolution: Media-resolution option supplied for multimodal requests.
			context: Prior conversation context supplied as dictionaries or Gemini Content objects.
			content: Optional text block prepended to the active prompt.
			urls: URLs to append to the provider request content.
			max_urls: Maximum number of URLs retained for request content.
			response_schema: Structured-output schema supplied as a dictionary, JSON string,
			or schema object.
			safety_profile: Named safety profile used to build Gemini safety settings.
			stream: Indicates whether streaming generation is requested.
			stream_handler: Optional callback used to receive streamed text chunks.

		Returns:
			str | None: Text output when available; otherwise None.

		Raises:
			Error: Raised when validation, provider execution, storage access, or response
			processing fails.
		"""
		try:
			throw_if( 'prompt', prompt )
			self.model = str( model or self.model or 'gemini-2.5-flash-lite' ).strip( )
			self.stream = bool( stream )
			self.urls = self._build_urls( urls=urls, max_urls=max_urls )
			self.content_block = self._append_urls_to_content( content=content, urls=self.urls )
			self.contents = self._build_contents(
				prompt=prompt,
				context=context,
				content=self.content_block )
			self.content_config = self._build_config(
				model=self.model,
				number=number,
				temperature=temperature,
				top_p=top_p,
				top_k=top_k,
				frequency=frequency,
				presence=presence,
				max_tokens=max_tokens,
				stops=stops,
				instruct=instruct,
				response_format=response_format,
				tools=tools,
				tool_choice=tool_choice,
				reasoning=reasoning,
				modalities=modalities,
				media_resolution=media_resolution,
				response_schema=response_schema,
				safety_profile=safety_profile )
			
			self.api_key = self._resolve_api_key( )
			throw_if( 'api_key', self.api_key )
			self.client = genai.Client( api_key=self.api_key )
			
			if self.stream:
				self.stream_response = self.client.models.generate_content_stream(
					model=self.model,
					contents=self.contents,
					config=self.content_config )
				
				if stream_handler is not None:
					self.text_blocks = [ ]
					for chunk in self.stream_response:
						if chunk is None:
							continue
						
						self.chunk_text = getattr( chunk, 'text', None )
						if self.chunk_text is None or not str( self.chunk_text ):
							continue
						
						self.text_blocks.append( str( self.chunk_text ) )
						stream_handler( str( self.chunk_text ) )
					
					self.output_text = ''.join( self.text_blocks ).strip( )
					return self.output_text if self.output_text else None
				
				return self.get_stream_output_text( stream_response=self.stream_response )
			
			self.content_response = self.client.models.generate_content(
				model=self.model,
				contents=self.contents,
				config=self.content_config )
			
			return self.get_output_text( )
		except Exception as e:
			exception = Error( e )
			exception.module = 'gemini'
			exception.cause = 'Chat'
			exception.method = 'generate_text( self, prompt, model ) -> Optional[ str ]'
			Logger( ).write( exception )
			raise exception

class Images( Gemini ):
	"""Define Images provider workflow state.

	Purpose:
		Coordinates Gemini image generation, image analysis, image editing, and image-grounding
		workflows. The class stores image request settings, builds image content configuration,
		opens local image inputs, captures grounding metadata, and extracts image or text outputs
		from Gemini responses.

	Attributes:
		client: Provider client used to execute Gemini requests.
		aspect_ratio: Image aspect ratio requested from the provider.
		use_vertex: Indicates whether Vertex AI behavior is enabled for compatible workflows.
		resolution: Image or media resolution requested from the provider.
		size: Image-size setting requested from supported image models.
	"""
	client: Optional[ genai.Client ]
	aspect_ratio: Optional[ str ]
	use_vertex: Optional[ bool ]
	resolution: Optional[ str ]
	size: Optional[ str ]
	
	def __init__( self, model: str = 'gemini-2.5-flash-image' ) -> None:
		"""Initialize Images runtime state.

		Purpose:
			Initializes Images runtime attributes without executing provider requests. The
			constructor prepares configuration fields, client placeholders, request state,
			response holders, and workflow-specific caches used by later methods.

		Args:
			model: Gemini model identifier used for the provider request.
		"""
		super( ).__init__( )
		self.number = None
		self.model = model
		self.client = None
		self.instructions = None
		self.content_config = None
		self.image_config = None
		self.function_config = None
		self.thought_config = None
		self.genimg_config = None
		self.tool_config = None
		self.response_modalities = [ ]
		self.tools = [ ]
		self.stops = [ ]
		self.domains = [ ]
		self.http_options = { }
		self.temperature = None
		self.size = None
		self.top_p = None
		self.top_k = None
		self.aspect_ratio = None
		self.frequency_penalty = None
		self.presence_penalty = None
		self.candidate_count = None
		self.max_output_tokens = None
		self.use_vertex = None
		self.media_resolution = None
		self.tool_choice = None
		self.content_response = None
		self.response = None
		self.grounding_metadata = None
		self.output_mime_type = None
		self.response_mode = None
	
	@property
	def model_options( self ) -> List[ str ] | None:
		"""Model options.

		Purpose:
			Returns the model identifiers exposed by this wrapper for user-interface selectors and
			request configuration.

		Returns:
			List[ str ] | None: List of string values when available; otherwise None.
		"""
		return [ 'gemini-2.5-flash-image',
		         'gemini-3.1-flash-image-preview' ]
	
	@property
	def include_options( self ) -> List[ str ] | None:
		"""Include options.

		Purpose:
			Returns include-field option names exposed by this wrapper for response expansion and
			diagnostic request configuration.

		Returns:
			List[ str ] | None: List of string values when available; otherwise None.
		"""
		return [ 'file_search_call.results',
		         'message.input_image.image_url',
		         'message.output_text.logprobs',
		         'reasoning.encrypted_content' ]
	
	@property
	def aspect_options( self ) -> List[ str ] | None:
		"""Aspect options.

		Purpose:
			Returns supported image aspect-ratio values for Gemini image-generation and
			image-editing requests.

		Returns:
			List[ str ] | None: List of string values when available; otherwise None.
		"""
		return [ '1:1', '2:3', '3:2', '3:4', '4:3', '4:5', '5:4', '9:16', '16:9', '21:9' ]
	
	@property
	def media_options( self ) -> List[ str ] | None:
		"""Media options.

		Purpose:
			Returns media-resolution option names exposed by this wrapper for multimodal request
			configuration.

		Returns:
			List[ str ] | None: List of string values when available; otherwise None.
		"""
		return [ 'media_resolution_high',
		         'media_resolution_medium',
		         'media_resolution_low' ]
	
	@property
	def modality_options( self ) -> List[ str ] | None:
		"""Modality options.

		Purpose:
			Returns response-modality option names exposed by this wrapper for text, image,
			and audio request configuration.

		Returns:
			List[ str ] | None: List of string values when available; otherwise None.
		"""
		return [ 'TEXT', 'IMAGE', 'TEXT_AND_IMAGE' ]
	
	@property
	def reasoning_options( self ) -> List[ str ] | None:
		"""Reasoning options.

		Purpose:
			Returns the reasoning-level names exposed by this wrapper for user-interface selectors
			and thinking configuration.

		Returns:
			List[ str ] | None: List of string values when available; otherwise None.
		"""
		return [ 'THINKING_LEVEL_UNSPECIFIED', 'MINIMAL',
		         'LOW', 'MEDIUM', 'HIGH' ]
	
	@property
	def size_options( self ) -> List[ str ] | None:
		"""Size options.

		Purpose:
			Returns supported generated-image size labels for models that expose image-size
			controls.

		Returns:
			List[ str ] | None: List of string values when available; otherwise None.
		"""
		return [ '1K', '2K', '4K' ]
	
	@property
	def tool_options( self ) -> List[ str ] | None:
		"""Tool options.

		Purpose:
			Returns the tool names exposed by this wrapper for user-interface selectors and
			provider request configuration.

		Returns:
			List[ str ] | None: List of string values when available; otherwise None.
		"""
		return [ 'google_search', 'image_search' ]
	
	@property
	def choice_options( self ) -> List[ str ] | None:
		"""Choice options.

		Purpose:
			Returns tool-choice option names exposed by this wrapper for provider tool-selection
			configuration.

		Returns:
			List[ str ] | None: List of string values when available; otherwise None.
		"""
		return [ 'AUTO', 'ANY', 'NONE', 'VALIDATED' ]
	
	@property
	def format_options( self ) -> List[ str ] | None:
		"""Format options.

		Purpose:
			Returns response-format option names exposed by this wrapper for MIME-type or
			content-format selection.

		Returns:
			List[ str ] | None: List of string values when available; otherwise None.
		"""
		return [ 'text/plain',
		         'application/json',
		         'text/x.enum' ]
	
	@property
	def mime_options( self ) -> List[ str ] | None:
		"""Mime options.

		Purpose:
			Returns output image MIME types exposed by the image wrapper for image generation and
			editing workflows.

		Returns:
			List[ str ] | None: List of string values when available; otherwise None.
		"""
		return [ 'image/jpeg',
		         'image/png',
		         'image/webp' ]
	
	@property
	def resolution_options( self ) -> List[ str ] | None:
		"""Resolution options.

		Purpose:
			Returns image-resolution option names exposed by the image wrapper for provider
			request configuration.

		Returns:
			List[ str ] | None: List of string values when available; otherwise None.
		"""
		return [ '1K', '2K', '4K' ]
	
	def _supports_image_size( self, model: str = None ) -> bool:
		"""Supports image size.

		Purpose:
			Determines whether the selected image model supports explicit image-size configuration.

		Args:
			model: Gemini model identifier used for the provider request.

		Returns:
			bool: Boolean result produced by the workflow.

		Raises:
			Error: Raised when validation, provider execution, storage access, or response
processing fails.
		"""
		try:
			self.model_name = str( model or self.model ).strip( )
			return self.model_name in [ 'gemini-3.1-flash-image-preview' ]
		except Exception as e:
			exception = Error( e )
			exception.module = 'gemini'
			exception.cause = 'Images'
			exception.method = '_supports_image_size( self, model=None ) -> bool'
			Logger( ).write( exception )
			raise exception
	
	def _supports_search_grounding( self, model: str = None ) -> bool:
		"""Supports search grounding.

		Purpose:
			Determines whether the selected image model supports Google Search grounding for image
			workflows.

		Args:
			model: Gemini model identifier used for the provider request.

		Returns:
			bool: Boolean result produced by the workflow.

		Raises:
			Error: Raised when validation, provider execution, storage access, or response
processing fails.
		"""
		try:
			self.model_name = str( model or self.model ).strip( )
			return self.model_name in [ 'gemini-3.1-flash-image-preview' ]
		except Exception as e:
			exception = Error( e )
			exception.module = 'gemini'
			exception.cause = 'Images'
			exception.method = '_supports_search_grounding( self, model=None ) -> bool'
			Logger( ).write( exception )
			raise exception
	
	def _supports_image_search( self, model: str = None ) -> bool:
		"""Supports image search.

		Purpose:
			Determines whether the selected image model supports Google Image Search grounding.

		Args:
			model: Gemini model identifier used for the provider request.

		Returns:
			bool: Boolean result produced by the workflow.

		Raises:
			Error: Raised when validation, provider execution, storage access, or response
			processing fails.
		"""
		try:
			self.model_name = str( model or self.model ).strip( )
			return self.model_name == 'gemini-3.1-flash-image-preview'
		except Exception as e:
			exception = Error( e )
			exception.module = 'gemini'
			exception.cause = 'Images'
			exception.method = '_supports_image_search( self, model=None ) -> bool'
			Logger( ).write( exception )
			raise exception
	
	def _normalize_response_modalities( self, response_modalities: Optional[ str ],
			image_only: bool = False ) -> List[ str ]:
		"""Normalize response modalities.

		Purpose:
			Normalizes a user-interface response-mode selection into Gemini response modalities.
			The method supports text, image, combined text-and-image, and workflow-specific
			defaults.

		Args:
			response_modalities: User-interface response-mode value normalized into Gemini
			modalities.
			image_only: Image Only value supplied to the workflow.

		Returns:
			List[ str ]: List of string values produced by the workflow.

		Raises:
			Error: Raised when validation, provider execution, storage access, or response
processing fails.
		"""
		try:
			self.mode_name = str( response_modalities or '' ).strip( ).upper( )
			
			if self.mode_name == 'TEXT_AND_IMAGE':
				return [ 'TEXT', 'IMAGE' ]
			
			if self.mode_name == 'TEXT':
				return [ 'TEXT' ]
			
			if self.mode_name == 'IMAGE':
				return [ 'IMAGE' ]
			
			return [ 'IMAGE' ] if image_only else [ 'TEXT' ]
		except Exception as e:
			exception = Error( e )
			exception.module = 'gemini'
			exception.cause = 'Images'
			exception.method = ('_normalize_response_modalities( self, response_modalities=None, '
			                    'image_only=False ) -> List[ str ]')
			Logger( ).write( exception )
			raise exception
	
	def _build_grounding_tool( self, image_search: bool = False ) -> Optional[ Tool ]:
		"""Build grounding tool.

		Purpose:
			Builds a Google Search grounding tool for supported image models. The method
			optionally requests image search when the selected model and SDK surface support it.

		Args:
			image_search: Indicates whether image-search grounding should be enabled when
			supported.

		Returns:
			Optional[ Tool ]: Gemini tool object when available; otherwise None.

		Raises:
			Error: Raised when validation, provider execution, storage access, or response
			processing fails.
		"""
		try:
			if not self._supports_search_grounding( self.model ):
				return None
			
			self.use_image_search = bool( image_search )
			self.model_name = str( self.model ).strip( )
			
			if self.use_image_search and self._supports_image_search( self.model_name ):
				try:
					return Tool(
						google_search=types.GoogleSearch(
							search_types=types.SearchTypes(
								web_search=types.WebSearch( ),
								image_search=types.ImageSearch( )
							)
						)
					)
				except Exception as e:
					ex = Error( e )
					ex.module = 'gemini'
					ex.cause = 'Images'
					ex.method = '_build_grounding_tool( self, *args )'
					Logger( ).write( ex )
					return Tool( google_search=types.GoogleSearch( ) )
			
			return Tool( google_search=types.GoogleSearch( ) )
		except Exception as e:
			exception = Error( e )
			exception.module = 'gemini'
			exception.cause = 'Images'
			exception.method = ('_build_grounding_tool( self, image_search=False ) -> Optional[ '
			                    'Tool ]')
			Logger( ).write( exception )
			raise exception
	
	def _get_content_config( self, image_only: bool = False, grounded: bool = False,
			image_search: bool = False, response_modalities: Optional[ str ] = None,
			output_mime_type: Optional[ str ] = None ) -> GenerateContentConfig:
		"""Get content config.

		Purpose:
			Builds GenerateContentConfig settings for image generation, analysis, and editing
			workflows. The method combines image configuration, grounding tools, output MIME type,
			modalities, and generation controls.

		Args:
			image_only: Image Only value supplied to the workflow.
			grounded: Indicates whether grounding tools should be enabled when supported.
			image_search: Indicates whether image-search grounding should be enabled when
			supported.
			response_modalities: User-interface response-mode value normalized into Gemini
			modalities.
			output_mime_type: Output image MIME type requested from Gemini image workflows.

		Returns:
			GenerateContentConfig: Provider-ready content generation configuration.

		Raises:
			Error: Raised when validation, provider execution, storage access, or response
			processing fails.
		"""
		try:
			self.image_config = None
			self.tool_config = None
			self.grounding_metadata = None
			self.image_kwargs = { }
			if self.aspect_ratio:
				self.image_kwargs[ 'aspect_ratio' ] = self.aspect_ratio
			
			if self.size and self._supports_image_size( self.model ):
				self.image_kwargs[ 'image_size' ] = self.size
			
			if output_mime_type:
				self.image_kwargs[ 'output_mime_type' ] = output_mime_type
			
			if len( self.image_kwargs ) > 0:
				self.image_config = types.ImageConfig( **self.image_kwargs )
			
			if grounded:
				self.grounding_tool = self._build_grounding_tool( image_search=image_search )
				if self.grounding_tool is not None:
					self.tool_config = [ self.grounding_tool ]
			
			self.response_modalities = self._normalize_response_modalities(
				response_modalities=response_modalities,
				image_only=image_only )
			
			self.config_kwargs = {
					'temperature': self.temperature,
					'top_p': self.top_p,
					'candidate_count': self.number,
					'max_output_tokens': self.max_output_tokens,
					'system_instruction': self.instructions,
					'response_modalities': self.response_modalities
			}
			
			if self.image_config is not None:
				self.config_kwargs[ 'image_config' ] = self.image_config
			
			if self.tool_config is not None and len( self.tool_config ) > 0:
				self.config_kwargs[ 'tools' ] = self.tool_config
			
			self.content_config = GenerateContentConfig( **self.config_kwargs )
			return self.content_config
		except Exception as e:
			exception = Error( e )
			exception.module = 'gemini'
			exception.cause = 'Images'
			exception.method = ('_get_content_config( self, image_only=False, grounded=False, '
			                    'image_search=False, response_modalities=None, '
			                    'output_mime_type=None ) -> GenerateContentConfig')
			Logger( ).write( exception )
			raise exception
	
	def _open_image( self, path: str ) -> PIL.Image.Image:
		"""Open image.

		Purpose:
			Opens a local image and returns a copied PIL image object for multimodal Gemini
			requests. Copying the image allows the source file handle to close immediately.

		Args:
			path: Local file path used by image, audio, or cloud-storage workflows.

		Returns:
			PIL.Image.Image: PIL image object produced by the workflow.

		Raises:
			Error: Raised when validation, provider execution, storage access, or response
			processing fails.
		"""
		try:
			throw_if( 'path', path )
			with PIL.Image.open( path ) as source:
				return source.copy( )
		except Exception as e:
			exception = Error( e )
			exception.module = 'gemini'
			exception.cause = 'Images'
			exception.method = '_open_image( self, path ) -> PIL.Image.Image'
			Logger( ).write( exception )
			raise exception
	
	def _capture_grounding_metadata( self ) -> None:
		"""Capture grounding metadata.

		Purpose:
			Captures grounding metadata from the most recent Gemini content response. The method
			stores metadata for downstream inspection while preserving a safe None value when no
			metadata is available.
		"""
		try:
			self.grounding_metadata = None
			if self.content_response is None:
				return
			
			self.candidates = getattr( self.content_response, 'candidates', None )
			if self.candidates:
				for candidate in self.candidates:
					self.metadata = getattr( candidate, 'grounding_metadata', None )
					if self.metadata is None:
						self.metadata = getattr( candidate, 'groundingMetadata', None )
					
					if self.metadata is not None:
						self.grounding_metadata = self.metadata
						return
		except Exception as e:
			ex = Error( e )
			ex.module = 'gemini'
			ex.cause = 'Images'
			ex.method = '_capture_grounding_metadata( self, *args )'
			Logger( ).write( ex )
			self.grounding_metadata = None
	
	def _get_first_image( self ) -> Optional[ PIL.Image.Image ]:
		"""Get first image.

		Purpose:
			Extracts the first image returned in a Gemini content response. The method checks
			top-level parts and candidate content parts while tolerating parts that cannot be
			converted to images.

		Returns:
			Optional[ PIL.Image.Image ]: PIL image output when available; otherwise None.

		Raises:
			Error: Raised when validation, provider execution, storage access, or response
			processing fails.
		"""
		try:
			if self.content_response is None:
				return None
			
			parts = getattr( self.content_response, 'parts', None )
			if parts:
				for part in parts:
					try:
						if getattr( part, 'inline_data', None ) is not None:
							return part.as_image( )
					except Exception as e:
						ex = Error( e )
						ex.module = 'gemini'
						ex.cause = 'Images'
						ex.method = '_get_first_image( self, *args )'
						Logger( ).write( ex )
						continue
			
			candidates = getattr( self.content_response, 'candidates', None )
			if candidates:
				for candidate in candidates:
					content = getattr( candidate, 'content', None )
					if content is None:
						continue
					
					candidate_parts = getattr( content, 'parts', None ) or [ ]
					for part in candidate_parts:
						try:
							if getattr( part, 'inline_data', None ) is not None:
								return part.as_image( )
						except Exception as e:
							ex = Error( e )
							ex.module = 'gemini'
							ex.cause = 'Images'
							ex.method = '_get_first_image( self, *args )'
							Logger( ).write( ex )
							continue
			
			return None
		except Exception as e:
			exception = Error( e )
			exception.module = 'gemini'
			exception.cause = 'Images'
			exception.method = '_get_first_image( self ) -> Optional[ PIL.Image.Image ]'
			Logger( ).write( exception )
			raise exception
	
	def _get_output_text( self ) -> Optional[ str ]:
		"""Get output text.

		Purpose:
			Extracts text output from a Gemini content response. The method checks response text,
			top-level parts, and candidate parts to support multiple SDK response shapes.

		Returns:
			Optional[ str ]: Text output when available; otherwise None.

		Raises:
			Error: Raised when validation, provider execution, storage access, or response
			processing fails.
		"""
		try:
			if self.content_response is None:
				return None
			
			text = getattr( self.content_response, 'text', None )
			if isinstance( text, str ) and text.strip( ):
				return text
			
			parts = getattr( self.content_response, 'parts', None )
			if parts:
				output = [ ]
				for part in parts:
					part_text = getattr( part, 'text', None )
					if isinstance( part_text, str ) and part_text.strip( ):
						output.append( part_text.strip( ) )
				
				if output:
					return '\n'.join( output )
			
			candidates = getattr( self.content_response, 'candidates', None )
			if candidates:
				for candidate in candidates:
					content = getattr( candidate, 'content', None )
					if content is None:
						continue
					
					output = [ ]
					for part in getattr( content, 'parts', None ) or [ ]:
						part_text = getattr( part, 'text', None )
						if isinstance( part_text, str ) and part_text.strip( ):
							output.append( part_text.strip( ) )
					
					if output:
						return '\n'.join( output )
			
			return None
		except Exception as e:
			exception = Error( e )
			exception.module = 'gemini'
			exception.cause = 'Images'
			exception.method = '_get_output_text( self ) -> Optional[ str ]'
			Logger( ).write( exception )
			raise exception
	
	def generate( self, prompt: str, model: str = 'gemini-2.5-flash-image', aspect: str = None,
			number: int = None, temperature: float = None, top_p: float = None,
			frequency: float = None, presence: float = None, max_tokens: int = None,
			resolution: str = None, instruct: str = None, output_mime_type: str = None,
			response_modalities: str = None, grounded: bool = False,
			image_search: bool = False ) -> Optional[ PIL.Image.Image ]:
		"""Generate an image with Gemini.

		Purpose:
			Generates an image from a text prompt using Gemini image models. The method validates
			the prompt, prepares image configuration, executes the provider request, captures
			grounding metadata, and returns the first generated image.

		Args:
			prompt: User prompt, instruction, or query submitted to the provider workflow.
			model: Gemini model identifier used for the provider request.
			aspect: Image aspect-ratio value used for generation or editing.
			number: Candidate count or output count requested from the provider.
			temperature: Sampling temperature applied to generation when supplied.
			top_p: Nucleus sampling threshold applied to generation when supplied.
			frequency: Frequency penalty applied to repeated tokens when supplied.
			presence: Presence penalty applied to repeated topics or tokens when supplied.
			max_tokens: Maximum output token count supplied to provider configuration.
			resolution: Image size or media-resolution value supplied to image workflows.
			instruct: Optional system instruction supplied to provider configuration.
			output_mime_type: Output image MIME type requested from Gemini image workflows.
			response_modalities: User-interface response-mode value normalized into Gemini
			modalities.
			grounded: Indicates whether grounding tools should be enabled when supported.
			image_search: Indicates whether image-search grounding should be enabled when
			supported.

		Returns:
			Optional[ PIL.Image.Image ]: PIL image output when available; otherwise None.

		Raises:
			Error: Raised when validation, provider execution, storage access, or response
			processing fails.
		"""
		try:
			throw_if( 'prompt', prompt )
			self.prompt = prompt
			self.model = model
			self.number = number
			self.aspect_ratio = aspect
			self.size = resolution
			self.top_p = top_p
			self.temperature = temperature
			self.frequency_penalty = frequency
			self.presence_penalty = presence
			self.max_output_tokens = max_tokens
			self.instructions = instruct
			self.output_mime_type = output_mime_type
			self.response_mode = response_modalities
			self.client = genai.Client( api_key=self.gemini_api_key )
			self.content_config = self._get_content_config(
				image_only=True,
				grounded=grounded,
				image_search=image_search,
				response_modalities=self.response_mode,
				output_mime_type=self.output_mime_type
			)
			self.content_response = self.client.models.generate_content(
				model=self.model,
				contents=[ self.prompt ],
				config=self.content_config
			)
			self.response = self.content_response
			self._capture_grounding_metadata( )
			return self._get_first_image( )
		except Exception as e:
			exception = Error( e )
			exception.module = 'gemini'
			exception.cause = 'Images'
			exception.method = 'generate( self, prompt, aspect ) -> Optional[ PIL.Image.Image ]'
			Logger( ).write( exception )
			raise exception
	
	def analyze( self, prompt: str, path: str, model: str = 'gemini-2.5-flash-image',
			aspect: str = None, number: int = None, temperature: float = None,
			top_p: float = None, frequency: float = None, presence: float = None,
			max_tokens: int = None, resolution: str = None, instruct: str = None,
			output_mime_type: str = None, response_modalities: str = None,
			grounded: bool = False, image_search: bool = False ) -> Optional[ str ]:
		"""Analyze an image with Gemini.

		Purpose:
			Analyzes a local image with a text prompt using Gemini multimodal input. The method
			validates prompt and path values, opens the image, executes the provider request,
			captures grounding metadata, and returns text analysis output.

		Args:
			prompt: User prompt, instruction, or query submitted to the provider workflow.
			path: Local file path used by image, audio, or cloud-storage workflows.
			model: Gemini model identifier used for the provider request.
			aspect: Image aspect-ratio value used for generation or editing.
			number: Candidate count or output count requested from the provider.
			temperature: Sampling temperature applied to generation when supplied.
			top_p: Nucleus sampling threshold applied to generation when supplied.
			frequency: Frequency penalty applied to repeated tokens when supplied.
			presence: Presence penalty applied to repeated topics or tokens when supplied.
			max_tokens: Maximum output token count supplied to provider configuration.
			resolution: Image size or media-resolution value supplied to image workflows.
			instruct: Optional system instruction supplied to provider configuration.
			output_mime_type: Output image MIME type requested from Gemini image workflows.
			response_modalities: User-interface response-mode value normalized into Gemini
			modalities.
			grounded: Indicates whether grounding tools should be enabled when supported.
			image_search: Indicates whether image-search grounding should be enabled when
			supported.

		Returns:
			Optional[ str ]: Text output when available; otherwise None.

		Raises:
			Error: Raised when validation, provider execution, storage access, or response
processing fails.
		"""
		try:
			throw_if( 'prompt', prompt )
			throw_if( 'path', path )
			self.prompt = prompt
			self.model = model
			self.number = number
			self.aspect_ratio = aspect
			self.media_resolution = resolution
			self.top_p = top_p
			self.temperature = temperature
			self.frequency_penalty = frequency
			self.presence_penalty = presence
			self.max_output_tokens = max_tokens
			self.instructions = instruct
			self.output_mime_type = output_mime_type
			self.response_mode = response_modalities or 'TEXT'
			self.client = genai.Client( api_key=self.gemini_api_key )
			self.content_config = self._get_content_config(
				image_only=False,
				grounded=grounded,
				image_search=image_search,
				response_modalities=self.response_mode,
				output_mime_type=self.output_mime_type
			)
			self.content_response = self.client.models.generate_content(
				model=self.model,
				contents=[ self.prompt, self._open_image( path ) ],
				config=self.content_config
			)
			self.response = self.content_response
			self._capture_grounding_metadata( )
			return self._get_output_text( )
		except Exception as e:
			exception = Error( e )
			exception.module = 'gemini'
			exception.cause = 'Images'
			exception.method = 'analyze( self, prompt, path, model ) -> Optional[ str ]'
			Logger( ).write( exception )
			raise exception
	
	def edit( self, prompt: str, path: str, model: str = 'gemini-2.5-flash-image',
			aspect: str = None, number: int = None, temperature: float = None,
			top_p: float = None, frequency: float = None, presence: float = None,
			max_tokens: int = None, resolution: str = None, instruct: str = None,
			output_mime_type: str = None, response_modalities: str = None,
			grounded: bool = False, image_search: bool = False ) -> Optional[ PIL.Image.Image ]:
		"""Edit an image with Gemini.

		Purpose:
			Edits a local image with a text instruction using Gemini image-capable models. The
			method validates prompt and path values, builds image configuration, submits the
			multimodal request, and returns the first edited image.

		Args:
			prompt: User prompt, instruction, or query submitted to the provider workflow.
			path: Local file path used by image, audio, or cloud-storage workflows.
			model: Gemini model identifier used for the provider request.
			aspect: Image aspect-ratio value used for generation or editing.
			number: Candidate count or output count requested from the provider.
			temperature: Sampling temperature applied to generation when supplied.
			top_p: Nucleus sampling threshold applied to generation when supplied.
			frequency: Frequency penalty applied to repeated tokens when supplied.
			presence: Presence penalty applied to repeated topics or tokens when supplied.
			max_tokens: Maximum output token count supplied to provider configuration.
			resolution: Image size or media-resolution value supplied to image workflows.
			instruct: Optional system instruction supplied to provider configuration.
			output_mime_type: Output image MIME type requested from Gemini image workflows.
			response_modalities: User-interface response-mode value normalized into Gemini
			modalities.
			grounded: Indicates whether grounding tools should be enabled when supported.
			image_search: Indicates whether image-search grounding should be enabled when
			supported.

		Returns:
			Optional[ PIL.Image.Image ]: PIL image output when available; otherwise None.

		Raises:
			Error: Raised when validation, provider execution, storage access, or response
 processing fails.
		"""
		try:
			throw_if( 'prompt', prompt )
			throw_if( 'path', path )
			self.prompt = prompt
			self.model = model
			self.number = number
			self.aspect_ratio = aspect
			self.size = resolution
			self.top_p = top_p
			self.temperature = temperature
			self.frequency_penalty = frequency
			self.presence_penalty = presence
			self.max_output_tokens = max_tokens
			self.instructions = instruct
			self.output_mime_type = output_mime_type
			self.response_mode = response_modalities
			self.client = genai.Client( api_key=self.gemini_api_key )
			self.content_config = self._get_content_config(
				image_only=True,
				grounded=grounded,
				image_search=image_search,
				response_modalities=self.response_mode,
				output_mime_type=self.output_mime_type
			)
			self.content_response = self.client.models.generate_content(
				model=self.model,
				contents=[ self.prompt, self._open_image( path ) ],
				config=self.content_config
			)
			self.response = self.content_response
			self._capture_grounding_metadata( )
			return self._get_first_image( )
		except Exception as e:
			exception = Error( e )
			exception.module = 'gemini'
			exception.cause = 'Images'
			exception.method = 'edit( self, prompt, path, model ) -> Optional[ PIL.Image.Image ]'
			Logger( ).write( exception )
			raise exception

class Embeddings( Gemini ):
	"""Define Embeddings provider workflow state.

	Purpose:
		Coordinates Gemini embedding generation for text inputs used by semantic retrieval and
		vector workflows. The class stores embedding model settings, builds EmbedContentConfig
		objects, sends text to the provider, and returns vector values for downstream search or
		similarity operations.

	Attributes:
		client: Provider client used to execute Gemini requests.
		response: Raw provider response captured by the active workflow.
		embedding: Embedding vector returned by the provider.
		encoding_format: Embedding response encoding format.
		dimensions: Requested embedding dimensionality.
		task_type: Embedding task type supplied to the provider.
		embedding_config: EmbedContentConfig used by embedding requests.
		contents: Content payload sent to Gemini.
		input_text: Input text processed by the active text, embedding, or speech workflow.
		file_path: Local file path used by file, image, audio, or document workflows.
		response_modalities: Output modalities requested from Gemini.
	"""
	client: Optional[ genai.Client ]
	response: Optional[ Any ]
	embedding: Optional[ List[ float ] ]
	encoding_format: Optional[ str ]
	dimensions: Optional[ int ]
	task_type: Optional[ str ]
	embedding_config: Optional[ types.EmbedContentConfig ]
	contents: Optional[ List[ str ] ]
	input_text: Optional[ str ]
	file_path: Optional[ str ]
	response_modalities: Optional[ str ]
	
	def __init__( self, model: str = 'gemini-embedding-001' ) -> None:
		"""Initialize Embeddings runtime state.

		Purpose:
			Initializes Embeddings runtime attributes without executing provider requests. The
			constructor prepares configuration fields, client placeholders, request state,
			response holders, and workflow-specific caches used by later methods.

		Args:
			model: Gemini model identifier used for the provider request.
		"""
		super( ).__init__( )
		self.model = model
		self.temperature = None
		self.top_p = None
		self.frequency_penalty = None
		self.presence_penalty = None
		self.max_tokens = None
		self.client = None
		self.embedding = None;
		self.response = None
		self.encoding_format = None
		self.input_text = None
		self.file_path = None
		self.dimensions = None
		self.task_type = None
		self.response_modalities = None
		self.embedding_config = None;
		self.content_config = None
	
	@property
	def model_options( self ) -> List[ str ] | None:
		"""Model options.

		Purpose:
			Returns the model identifiers exposed by this wrapper for user-interface selectors and
			request configuration.

		Returns:
			List[ str ] | None: List of string values when available; otherwise None.
		"""
		return [ 'gemini-embedding-001',
		         'text-multilingual-embedding-002' ]
	
	@property
	def encoding_options( self ) -> List[ str ]:
		"""Encoding options.

		Purpose:
			Returns embedding encoding format names exposed by the embedding wrapper.

		Returns:
			List[ str ]: List of string values produced by the workflow.
		"""
		return [ 'float', 'base64' ]
	
	@property
	def task_options( self ) -> List[ str ]:
		"""Task options.

		Purpose:
			Returns embedding task-type names used to configure retrieval, semantic-similarity,
		classification, and clustering embeddings.

		Returns:
			List[ str ]: List of string values produced by the workflow.
		"""
		return [ 'RETRIEVAL_QUERY', 'RETRIEVAL_DOCUMENT', 'SEMANTIC_SIMILARITY',
		         'CLASSIFICATION', 'CLUSTERING' ]
	
	def create( self, text: str, model: str = 'gemini-embedding-001', temperature: float = None,
			top_p: float = None, frequency: float = None, presence: float = None,
			max_tokens: int = None ) -> List[ float ] | None:
		"""Create or modify the provider resource represented by the wrapper.

		Purpose:
			Creates the provider resource or output represented by the active wrapper. Depending
			on the class, the method creates embeddings, remote objects, or cloud-storage entries
			while preserving wrapper state for follow-on operations.

		Args:
			text: Input text used for embedding or speech generation.
			model: Gemini model identifier used for the provider request.
			temperature: Sampling temperature applied to generation when supplied.
			top_p: Nucleus sampling threshold applied to generation when supplied.
			frequency: Frequency penalty applied to repeated tokens when supplied.
			presence: Presence penalty applied to repeated topics or tokens when supplied.
			max_tokens: Maximum output token count supplied to provider configuration.

		Returns:
			List[ float ] | None: Embedding vector values when available; otherwise None.

		Raises:
			Error: Raised when validation, provider execution, storage access, or response
 processing fails.
		"""
		try:
			throw_if( 'text', text )
			self.input_text = text;
			self.model = model
			self.temperature = temperature
			self.top_p = top_p
			self.frequency_penalty = frequency
			self.presence_penalty = presence
			self.max_tokens = max_tokens
			self.client = genai.Client( api_key=self.gemini_api_key )
			self.embedding_config = EmbedContentConfig( task_type=self.task_type )
			self.response = self.client.models.embed_content( model=self.model,
				contents=self.input_text, config=self.embedding_config )
			self.embedding = self.response.embeddings[ 0 ].values
			return self.embedding
		except Exception as e:
			exception = Error( e )
			exception.module = 'gemini'
			exception.cause = 'Embedding'
			exception.method = 'generate( self, text, model ) -> List[ float ]'
			Logger( ).write( exception )
			raise exception

class TTS( Gemini ):
	"""Define TTS provider workflow state.

	Purpose:
		Coordinates Gemini text-to-speech workflows that convert text into WAV audio. The class
		stores voice settings, builds speech configuration, extracts returned PCM audio,
		wraps audio bytes in a WAV container, and optionally writes generated audio to disk.

	Attributes:
		speed: Speech speed hint retained by the text-to-speech wrapper.
		voice: Prebuilt Gemini voice name used by text-to-speech.
		response: Raw provider response captured by the active workflow.
		voice_config: Voice configuration used by text-to-speech requests.
		speech_config: Speech configuration supplied to Gemini TTS.
		client: Provider client used to execute Gemini requests.
		audio_path: Optional output path for generated audio.
		response_format: Response MIME type or output format requested from the provider.
		input_text: Input text processed by the active text, embedding, or speech workflow.
		audio_bytes: Generated WAV bytes returned by text-to-speech.
	"""
	speed: Optional[ float ]
	voice: Optional[ str ]
	response: Optional[ GenerateContentResponse ]
	voice_config: Optional[ VoiceConfig ]
	speech_config: Optional[ SpeechConfig ]
	client: Optional[ genai.Client ]
	audio_path: Optional[ str ]
	response_format: Optional[ str ]
	input_text: Optional[ str ]
	audio_bytes: Optional[ bytes ]
	
	def __init__( self, model: str = 'gemini-2.5-flash-preview-tts' ) -> None:
		"""Initialize TTS runtime state.

		Purpose:
			Initializes TTS runtime attributes without executing provider requests. The
			constructor prepares configuration fields, client placeholders, request state,
			response holders, and workflow-specific caches used by later methods.

		Args:
			model: Gemini model identifier used for the provider request.
		"""
		super( ).__init__( )
		self.number = None
		self.model = model
		self.temperature = None
		self.top_p = None
		self.frequency_penalty = None
		self.presence_penalty = None
		self.max_tokens = None
		self.instructions = None
		self.voice_config = None
		self.speech_config = None
		self.content_config = None
		self.client = None
		self.voice = None
		self.speed = None
		self.response = None
		self.response_format = None
		self.audio_path = None
		self.input_text = None
		self.audio_bytes = None
		self.response_modalities = [ ]
	
	@property
	def model_options( self ) -> List[ str ] | None:
		"""Model options.

		Purpose:
			Returns the model identifiers exposed by this wrapper for user-interface selectors and
			request configuration.

		Returns:
			List[ str ] | None: List of string values when available; otherwise None.
		"""
		return [ 'gemini-2.5-flash-preview-tts',
		         'gemini-2.5-pro-preview-tts' ]
	
	@property
	def voice_options( self ) -> List[ str ] | None:
		"""Voice options.

		Purpose:
			Returns Gemini prebuilt voice names exposed by the text-to-speech wrapper.

		Returns:
			List[ str ] | None: List of string values when available; otherwise None.
		"""
		return [ 'Zephyr', 'Puck', 'Charon', 'Kore', 'Fenrir', 'Leda', 'Orus', 'Aoede',
		         'Callirhoe',
		         'Autonoe', 'Enceladus', 'Iapetus', 'Umbriel', 'Algieba', 'Despina', 'Erinome',
		         'Algenib', 'Rasalgethi', 'Laomedeia', 'Achernar', 'Alnilam', 'Schedar', 'Gacrux',
		         'Pulcherrima', 'Achird', 'Zubenelgenubi', 'Vindemiatrix', 'Sadachbia',
		         'Sadaltager', 'Sulafar' ]
	
	@property
	def format_options( self ) -> List[ str ] | None:
		"""Format options.

		Purpose:
			Returns response-format option names exposed by this wrapper for MIME-type or
			content-format selection.

		Returns:
			List[ str ] | None: List of string values when available; otherwise None.
		"""
		return [ 'audio/wav' ]
	
	def _to_wave_bytes( self, pcm_data: bytes, rate: int = 24000, channels: int = 1,
			sample_width: int = 2 ) -> bytes:
		"""To wave bytes.

		Purpose:
			Wraps raw PCM audio bytes returned by Gemini text-to-speech in a WAV container. The
			method writes sample metadata and frames into an in-memory wave file and returns the
			resulting bytes.

		Args:
			pcm_data: Raw PCM audio bytes returned by the provider.
			rate: Audio sample rate used when wrapping PCM data in a WAV container.
			channels: Audio channel count used when wrapping PCM data in a WAV container.
			sample_width: Sample width in bytes used when wrapping PCM data in a WAV container.

		Returns:
			bytes: Byte content produced by the workflow.
		"""
		import io
		import wave
		
		with io.BytesIO( ) as buffer:
			with wave.open( buffer, 'wb' ) as wf:
				wf.setnchannels( channels )
				wf.setsampwidth( sample_width )
				wf.setframerate( rate )
				wf.writeframes( pcm_data )
			
			return buffer.getvalue( )
	
	def create_speech( self, text: str, filepath: str = None,
			model: str = 'gemini-2.5-flash-preview-tts', format: str = 'audio/wav',
			speed: float = None, voice: str = None, frequency: float = None,
			presense: float = None, max_tokens: int = None, instruct: str = None,
			temperature: float = None, top_p: float = None ) -> bytes | str | None:
		"""Create speech audio with Gemini.

		Purpose:
			Converts text into speech using Gemini text-to-speech models. The method validates
			input, builds voice and speech configuration, extracts returned audio bytes, converts
			PCM data to WAV, and either writes the file or returns bytes.

		Args:
			text: Input text used for embedding or speech generation.
			filepath: Local file path used by file upload or document workflows.
			model: Gemini model identifier used for the provider request.
			format: Output format requested by the audio or response workflow.
			speed: Speed value supplied to the workflow.
			voice: Gemini prebuilt voice name used by the text-to-speech workflow.
			frequency: Frequency penalty applied to repeated tokens when supplied.
			presense: Presence penalty value retained for compatibility with the existing method
			signature.
			max_tokens: Maximum output token count supplied to provider configuration.
			instruct: Optional system instruction supplied to provider configuration.
			temperature: Sampling temperature applied to generation when supplied.
			top_p: Nucleus sampling threshold applied to generation when supplied.

		Returns:
			bytes | str | None: Generated audio bytes, output file path, or None.

		Raises:
			Error: Raised when validation, provider execution, storage access, or response
 processing fails.
		"""
		try:
			throw_if( 'text', text )
			self.input_text = text
			self.audio_path = filepath
			self.response_format = str( format or 'audio/wav' ).strip( )
			self.speed = speed
			self.voice = str( voice or 'Kore' ).strip( )
			self.max_tokens = max_tokens
			self.model = str( model or self.model or 'gemini-2.5-flash-preview-tts' ).strip( )
			self.frequency_penalty = frequency
			self.presence_penalty = presense
			self.instructions = instruct
			self.temperature = temperature
			self.top_p = top_p
			self.response_modalities = [ 'AUDIO' ]
			self.voice_config = VoiceConfig( prebuilt_voice_config=types.PrebuiltVoiceConfig(
				voice_name=self.voice ) )
			self.speech_config = SpeechConfig( voice_config=self.voice_config )
			self.config_kwargs = { 'response_modalities': self.response_modalities,
			                       'speech_config': self.speech_config }
			
			if self.temperature is not None:
				self.config_kwargs[ 'temperature' ] = self.temperature
			
			if self.top_p is not None:
				self.config_kwargs[ 'top_p' ] = self.top_p
			
			if self.max_tokens is not None:
				self.config_kwargs[ 'max_output_tokens' ] = self.max_tokens
			
			if self.instructions is not None and str( self.instructions ).strip( ):
				self.config_kwargs[ 'system_instruction' ] = str( self.instructions ).strip( )
			
			self.content_config = GenerateContentConfig( **self.config_kwargs )
			self.client = genai.Client( api_key=self.gemini_api_key )
			self.response = self.client.models.generate_content(
				model=self.model, contents=self.input_text,
				config=self.content_config )
			
			self.audio_bytes = None
			for part in self.response.candidates[ 0 ].content.parts:
				if getattr( part, 'inline_data', None ) is not None and part.inline_data.data:
					self.audio_bytes = self._to_wave_bytes( part.inline_data.data )
					break
			
			if self.audio_bytes is None:
				raise ValueError( 'No audio bytes were returned by Gemini TTS.' )
			
			if self.audio_path is not None and str( self.audio_path ).strip( ):
				with open( self.audio_path, 'wb' ) as f:
					f.write( self.audio_bytes )
				
				return self.audio_path
			
			return self.audio_bytes
		except Exception as e:
			exception = Error( e )
			exception.module = 'gemini'
			exception.cause = 'TTS'
			exception.method = 'create_speech( self, text, filepath, model, format, speed, voice )'
			Logger( ).write( exception )
			raise exception

class Transcription( Gemini ):
	"""Define Transcription provider workflow state.

	Purpose:
		Coordinates Gemini audio-transcription workflows. The class stores audio request settings,
		builds transcription prompts from language and time-range options, uploads audio files to
		the provider, and returns transcript text for Jimi audio-understanding operations.

	Attributes:
		client: Provider client used to execute Gemini requests.
		transcript: Transcript text returned by audio transcription.
		file_path: Local file path used by file, image, audio, or document workflows.
		response: Raw provider response captured by the active workflow.
	"""
	client: Optional[ genai.Client ]
	transcript: Optional[ str ]
	file_path: Optional[ str ]
	response: Optional[ GenerateContentResponse ]
	
	def __init__( self, n: int = 1, model: str = 'gemini-3-flash-preview', temperature: float =
	0.8,
			top_p: float = 0.9, frequency: float = 0.0, presence: float = 0.0,
			max_tokens: int = 10000, instruct: str = None ) -> None:
		"""Initialize Transcription runtime state.

		Purpose:
			Initializes Transcription runtime attributes without executing provider requests. The
			constructor prepares configuration fields, client placeholders, request state,
			response holders, and workflow-specific caches used by later methods.

		Args:
			n: N value supplied to the workflow.
			model: Gemini model identifier used for the provider request.
			temperature: Sampling temperature applied to generation when supplied.
			top_p: Nucleus sampling threshold applied to generation when supplied.
			frequency: Frequency penalty applied to repeated tokens when supplied.
			presence: Presence penalty applied to repeated topics or tokens when supplied.
			max_tokens: Maximum output token count supplied to provider configuration.
			instruct: Optional system instruction supplied to provider configuration.
		"""
		super( ).__init__( )
		self.number = n
		self.model = model
		self.temperature = temperature
		self.top_p = top_p
		self.frequency_penalty = frequency
		self.presence_penalty = presence
		self.max_tokens = max_tokens
		self.instructions = instruct
		self.client = genai.Client( api_key=self.gemini_api_key )
		self.transcript = None
		self.file_path = None
		self.response = None
		self.content_config = None
	
	@property
	def model_options( self ) -> List[ str ] | None:
		"""Model options.

		Purpose:
			Returns the model identifiers exposed by this wrapper for user-interface selectors and
			request configuration.

		Returns:
			List[ str ] | None: List of string values when available; otherwise None.
		"""
		return [ 'gemini-3-flash-preview',
		         'gemini-2.0-flash' ]
	
	@property
	def language_options( self ) -> List[ str ] | None:
		"""Language options.

		Purpose:
			Returns language labels exposed by audio transcription and translation workflows.

		Returns:
			List[ str ] | None: List of string values when available; otherwise None.
		"""
		return [ 'Auto',
		         'English',
		         'Spanish',
		         'French',
		         'Japanese',
		         'German',
		         'Chinese' ]
	
	@property
	def format_options( self ) -> List[ str ] | None:
		"""Format options.

		Purpose:
			Returns response-format option names exposed by this wrapper for MIME-type or
		         content-format selection.

		Returns:
			List[ str ] | None: List of string values when available; otherwise None.
		"""
		return [ 'audio/wav',
		         'audio/mp3',
		         'audio/x-m4a',
		         'audio/flac' ]
	
	def _build_prompt( self, language: str = None, start_time: float = None,
			end_time: float = None ) -> str:
		"""Build prompt.

		Purpose:
			Builds an audio workflow prompt from language and time-range settings. The method
			produces concise provider instructions used by transcription or translation requests.

		Args:
			language: Language hint or target language used by audio workflows.
			start_time: Optional start timestamp in seconds for audio slicing instructions.
			end_time: Optional end timestamp in seconds for audio slicing instructions.

		Returns:
			str: Text output produced by the workflow.
		"""
		self.prompt_parts = [ 'Generate a verbatim transcript of the speech.' ]
		
		if (language is not None and str( language ).strip( ) and str( language ).strip( ) !=
				'Auto'):
			self.prompt_parts.append(
				f'The expected spoken language is {str( language ).strip( )}.' )
		
		if start_time is not None and end_time is not None and end_time >= start_time:
			self.prompt_parts.append(
				f'Only transcribe the portion of the audio between {start_time:0.2f} seconds '
				f'and {end_time:0.2f} seconds.' )
		
		self.prompt_parts.append( 'Return only the transcript text.' )
		return ' '.join( self.prompt_parts )
	
	def transcribe( self, path: str, model: str = 'gemini-3-flash-preview',
			language: str = None, mime_type: str = None, temperature: float = None,
			top_p: float = None, frequency: float = None, presence: float = None,
			max_tokens: int = None, start_time: float = None, end_time: float = None,
			instruct: str = None ) -> Optional[ str ]:
		"""Transcribe audio with Gemini.

		Purpose:
			Transcribes a local audio file using Gemini audio understanding. The method validates
			the path, builds a transcription prompt, uploads the audio file, executes the provider
			request, and returns transcript text.

		Args:
			path: Local file path used by image, audio, or cloud-storage workflows.
			model: Gemini model identifier used for the provider request.
			language: Language hint or target language used by audio workflows.
			mime_type: MIME type hint supplied for audio upload workflows.
			temperature: Sampling temperature applied to generation when supplied.
			top_p: Nucleus sampling threshold applied to generation when supplied.
			frequency: Frequency penalty applied to repeated tokens when supplied.
			presence: Presence penalty applied to repeated topics or tokens when supplied.
			max_tokens: Maximum output token count supplied to provider configuration.
			start_time: Optional start timestamp in seconds for audio slicing instructions.
			end_time: Optional end timestamp in seconds for audio slicing instructions.
			instruct: Optional system instruction supplied to provider configuration.

		Returns:
			Optional[ str ]: Text output when available; otherwise None.

		Raises:
			Error: Raised when validation, provider execution, storage access, or response
			processing fails.
		"""
		try:
			import mimetypes
			
			throw_if( 'path', path )
			self.file_path = path
			self.model = str( model or self.model or 'gemini-3-flash-preview' ).strip( )
			self.temperature = temperature if temperature is not None else self.temperature
			self.top_p = top_p if top_p is not None else self.top_p
			self.frequency_penalty = frequency if frequency is not None else self.frequency_penalty
			self.presence_penalty = presence if presence is not None else self.presence_penalty
			self.max_tokens = max_tokens if max_tokens is not None else self.max_tokens
			self.instructions = instruct if instruct is not None else self.instructions
			self.mime_type = (mime_type or mimetypes.guess_type( self.file_path )[ 0 ] or
			                  'audio/wav')
			self.prompt = self._build_prompt( language=language, start_time=start_time,
				end_time=end_time )
			
			self.config_kwargs = { }
			
			if self.temperature is not None:
				self.config_kwargs[ 'temperature' ] = self.temperature
			
			if self.top_p is not None:
				self.config_kwargs[ 'top_p' ] = self.top_p
			
			if self.max_tokens is not None:
				self.config_kwargs[ 'max_output_tokens' ] = self.max_tokens
			
			if self.instructions is not None and str( self.instructions ).strip( ):
				self.config_kwargs[ 'system_instruction' ] = str( self.instructions ).strip( )
			
			self.content_config = GenerateContentConfig( **self.config_kwargs )
			self.uploaded_file = self.client.files.upload( file=self.file_path )
			self.response = self.client.models.generate_content(
				model=self.model,
				contents=[ self.prompt, self.uploaded_file ],
				config=self.content_config )
			self.transcript = self.response.text
			return self.transcript
		except Exception as e:
			ex = Error( e )
			ex.module = 'gemini'
			ex.cause = 'Transcription'
			ex.method = 'transcribe( self, path, model, language ) -> str'
			Logger( ).write( ex )
			raise ex

class Translation( Gemini ):
	"""Define Translation provider workflow state.

	Purpose:
		Coordinates Gemini audio-translation workflows. The class stores source and target
		language settings, builds translation prompts from language and time-range options,
		uploads audio files to the provider, and returns translated speech text.

	Attributes:
		client: Provider client used to execute Gemini requests.
		target_language: Target language used by audio translation.
		source_language: Source language hint used by audio translation.
		file_path: Local file path used by file, image, audio, or document workflows.
		response: Raw provider response captured by the active workflow.
	"""
	client: Optional[ genai.Client ]
	target_language: Optional[ str ]
	source_language: Optional[ str ]
	file_path: Optional[ str ]
	response: Optional[ GenerateContentResponse ]
	
	def __init__( self, n: int = 1, model: str = 'gemini-3-flash-preview', temperature: float =
	0.8,
			top_p: float = 0.9, frequency: float = 0.0, presence: float = 0.0,
			max_tokens: int = 10000,
			instruct: str = None ) -> None:
		"""Initialize Translation runtime state.

		Purpose:
			Initializes Translation runtime attributes without executing provider requests. The
			constructor prepares configuration fields, client placeholders, request state,
			response holders, and workflow-specific caches used by later methods.

		Args:
			n: N value supplied to the workflow.
			model: Gemini model identifier used for the provider request.
			temperature: Sampling temperature applied to generation when supplied.
			top_p: Nucleus sampling threshold applied to generation when supplied.
			frequency: Frequency penalty applied to repeated tokens when supplied.
			presence: Presence penalty applied to repeated topics or tokens when supplied.
			max_tokens: Maximum output token count supplied to provider configuration.
			instruct: Optional system instruction supplied to provider configuration.
		"""
		super( ).__init__( )
		self.number = n
		self.model = model
		self.temperature = temperature
		self.top_p = top_p
		self.frequency_penalty = frequency
		self.presence_penalty = presence
		self.max_tokens = max_tokens
		self.instructions = instruct
		self.client = genai.Client( api_key=self.gemini_api_key )
		self.target_language = None
		self.source_language = None
		self.file_path = None
		self.response = None
		self.content_config = None
	
	@property
	def model_options( self ) -> List[ str ] | None:
		"""Model options.

		Purpose:
			Returns the model identifiers exposed by this wrapper for user-interface selectors and
			request configuration.

		Returns:
			List[ str ] | None: List of string values when available; otherwise None.
		"""
		return [ 'gemini-3-flash-preview',
		         'gemini-2.0-flash' ]
	
	@property
	def format_options( self ) -> List[ str ] | None:
		"""Format options.

		Purpose:
			Returns response-format option names exposed by this wrapper for MIME-type or
			content-format selection.

		Returns:
			List[ str ] | None: List of string values when available; otherwise None.
		"""
		return [ 'audio/wav',
		         'audio/mp3',
		         'audio/x-m4a',
		         'audio/flac' ]
	
	@property
	def language_options( self ) -> List[ str ] | None:
		"""Language options.

		Purpose:
			Returns language labels exposed by audio transcription and translation workflows.

		Returns:
			List[ str ] | None: List of string values when available; otherwise None.
		"""
		return [ 'English',
		         'Spanish',
		         'French',
		         'Japanese',
		         'German',
		         'Chinese' ]
	
	def _build_prompt( self, target: str, source: str = 'Auto', start_time: float = None,
			end_time: float = None ) -> str:
		"""Build prompt.

		Purpose:
			Builds an audio workflow prompt from language and time-range settings. The method
			produces concise provider instructions used by transcription or translation requests.

		Args:
			target: Target translation language used to build the translation prompt.
			source: Source-language hint used by the translation workflow.
			start_time: Optional start timestamp in seconds for audio slicing instructions.
			end_time: Optional end timestamp in seconds for audio slicing instructions.

		Returns:
			str: Text output produced by the workflow.
		"""
		self.prompt_parts = [ f'Translate the spoken audio into {target}.' ]
		
		if source is not None and str( source ).strip( ) and str( source ).strip( ) != 'Auto':
			self.prompt_parts.append(
				f'The expected source language is {str( source ).strip( )}.' )
		
		if start_time is not None and end_time is not None and end_time >= start_time:
			self.prompt_parts.append(
				f'Only translate the portion of the audio between {start_time:0.2f} seconds '
				f'and {end_time:0.2f} seconds.' )
		
		self.prompt_parts.append( 'Return only the translated text.' )
		return ' '.join( self.prompt_parts )
	
	def translate( self, path: str, model: str = 'gemini-3-flash-preview',
			language: str = 'English', source: str = 'Auto', mime_type: str = None,
			temperature: float = None, top_p: float = None, frequency: float = None,
			presence: float = None, max_tokens: int = None, start_time: float = None,
			end_time: float = None, instruct: str = None ) -> Optional[ str ]:
		"""Translate audio with Gemini.

		Purpose:
			Translates spoken audio into a target language using Gemini audio understanding. The
			method validates the path, builds a translation prompt, uploads the audio file,
			executes the provider request, and returns translated text.

		Args:
			path: Local file path used by image, audio, or cloud-storage workflows.
			model: Gemini model identifier used for the provider request.
			language: Language hint or target language used by audio workflows.
			source: Source-language hint used by the translation workflow.
			mime_type: MIME type hint supplied for audio upload workflows.
			temperature: Sampling temperature applied to generation when supplied.
			top_p: Nucleus sampling threshold applied to generation when supplied.
			frequency: Frequency penalty applied to repeated tokens when supplied.
			presence: Presence penalty applied to repeated topics or tokens when supplied.
			max_tokens: Maximum output token count supplied to provider configuration.
			start_time: Optional start timestamp in seconds for audio slicing instructions.
			end_time: Optional end timestamp in seconds for audio slicing instructions.
			instruct: Optional system instruction supplied to provider configuration.

		Returns:
			Optional[ str ]: Text output when available; otherwise None.

		Raises:
			Error: Raised when validation, provider execution, storage access, or response
			processing fails.
		"""
		try:
			import mimetypes
			
			throw_if( 'path', path )
			self.file_path = path
			self.model = str( model or self.model or 'gemini-3-flash-preview' ).strip( )
			self.target_language = str( language or 'English' ).strip( )
			self.source_language = str( source or 'Auto' ).strip( )
			self.temperature = temperature if temperature is not None else self.temperature
			self.top_p = top_p if top_p is not None else self.top_p
			self.frequency_penalty = frequency if frequency is not None else self.frequency_penalty
			self.presence_penalty = presence if presence is not None else self.presence_penalty
			self.max_tokens = max_tokens if max_tokens is not None else self.max_tokens
			self.instructions = instruct if instruct is not None else self.instructions
			self.mime_type = (mime_type or mimetypes.guess_type( self.file_path )[ 0 ] or
			                  'audio/wav')
			self.prompt = self._build_prompt(
				target=self.target_language,
				source=self.source_language,
				start_time=start_time,
				end_time=end_time )
			
			self.config_kwargs = { }
			
			if self.temperature is not None:
				self.config_kwargs[ 'temperature' ] = self.temperature
			
			if self.top_p is not None:
				self.config_kwargs[ 'top_p' ] = self.top_p
			
			if self.max_tokens is not None:
				self.config_kwargs[ 'max_output_tokens' ] = self.max_tokens
			
			if self.instructions is not None and str( self.instructions ).strip( ):
				self.config_kwargs[ 'system_instruction' ] = str( self.instructions ).strip( )
			
			self.content_config = GenerateContentConfig( **self.config_kwargs )
			self.uploaded_file = self.client.files.upload( file=self.file_path )
			self.response = self.client.models.generate_content(
				model=self.model,
				contents=[ self.prompt, self.uploaded_file ],
				config=self.content_config )
			return self.response.text
		except Exception as e:
			ex = Error( e )
			ex.module = 'gemini'
			ex.cause = 'Translation'
			ex.method = 'translate( self, path, model, language, source ) -> str'
			Logger( ).write( ex )
			raise ex

class Files( Gemini ):
	"""Define Files provider workflow state.

	Purpose:
		Wraps Gemini file and document workflows used to upload, retrieve, summarize, search,
		survey, and delete remote assets. The class stores local file paths, remote identifiers,
		model settings, and provider responses used by document-oriented Gemini operations.

	Attributes:
		api_version: API version or HTTP option value used for provider clients.
		google_api_key: Google API key read from project configuration.
		storage_client: Google Cloud Storage client used by storage-backed workflows.
		project_id: Google Cloud project identifier used by storage workflows.
		project_location: Google Cloud location used by provider configuration.
		file_id: Remote file identifier used by Gemini file workflows.
		bucket_id: Remote bucket identifier used by storage workflows.
		display_name: Display name supplied for uploaded files.
		mime_type: MIME type assigned to an uploaded or analyzed file.
		file_path: Local file path used by file, image, audio, or document workflows.
		file_list: Remote file metadata collection.
		file_paths: Local file paths used by multi-document workflows.
		file_lists: Remote file-list state retained by the wrapper.
		response: Raw provider response captured by the active workflow.
		use_vertex: Indicates whether Vertex AI behavior is enabled for compatible workflows.
		collections: Named storage collections exposed by the wrapper.
		documents: Named remote documents or blob identifiers tracked by the wrapper.
	"""
	api_version: Optional[ str ]
	google_api_key: Optional[ str ]
	storage_client: Optional[ storage.Client ]
	project_id: Optional[ str ]
	project_location: Optional[ str ]
	file_id: Optional[ str ]
	bucket_id: Optional[ str ]
	display_name: Optional[ str ]
	mime_type: Optional[ str ]
	file_path: Optional[ str ]
	file_list: Optional[ List[ File ] ]
	file_paths: Optional[ List[ str ] ]
	file_lists: Optional[ List[ File ] ]
	response: Optional[ Any ]
	use_vertex: Optional[ bool ]
	collections: Optional[ Dict[ str, str ] ]
	documents: Optional[ Dict[ str, str ] ]
	
	def __init__( self, model: str = 'gemini-2.0-flash' ) -> None:
		"""Initialize Files runtime state.

		Purpose:
			Initializes Files runtime attributes without executing provider requests. The
			constructor prepares configuration fields, client placeholders, request state,
response holders, and workflow-specific caches used by later methods.

		Args:
			model: Gemini model identifier used for the provider request.
		"""
		super( ).__init__( )
		self.google_api_key = cfg.GOOGLE_API_KEY
		self.project_id = cfg.GOOGLE_CLOUD_PROJECT_ID
		self.project_location = cfg.GOOGLE_CLOUD_LOCATION
		self.model = model
		self.top_p = None
		self.top_k = None
		self.temperature = None
		self.frequency_penalty = None
		self.presence_penalty = None
		self.max_tokens = None
		self.tool_choice = None
		self.stops = [ ]
		self.response_modalities = [ ]
		self.tools = [ ]
		self.domains = [ ]
		self.http_options = { }
		self.storage_client = None
		self.bucket_id = None
		self.file_id = None
		self.display_name = None
		self.media_resolution = None
		self.mime_type = None
		self.file_path = None
		self.file_list = [ ]
		self.response = None
		self.collections = { }
		self.documents = { }
	
	@property
	def file_options( self ) -> List[ str ] | None:
		"""File options.

		Purpose:
			Returns available file identifiers tracked by the file wrapper for user-interface
	selectors and file-management workflows.

		Returns:
			List[ str ] | None: List of string values when available; otherwise None.
		"""
		return self.files
	
	@property
	def model_options( self ) -> List[ str ] | None:
		"""Model options.

		Purpose:
			Returns the model identifiers exposed by this wrapper for user-interface selectors and
			request configuration.

		Returns:
			List[ str ] | None: List of string values when available; otherwise None.
		"""
		return [ 'gemini-3.5-flash',
		         'gemini-3.5 flash-lite',
		         'gemini-3.0-flash',
		         'gemini-3.0-flash-lite' ]
	
	@property
	def media_options( self ) -> List[ str ] | None:
		"""Media options.

		Purpose:
			Returns media-resolution option names exposed by this wrapper for multimodal request
			configuration.

		Returns:
			List[ str ] | None: List of string values when available; otherwise None.
		"""
		return [ 'media_resolution_high',
		         'media_resolution_medium',
		         'media_resolution_low' ]
	
	@property
	def include_options( self ) -> List[ str ] | None:
		"""Include options.

		Purpose:
			Returns include-field option names exposed by this wrapper for response expansion and
		         diagnostic request configuration.

		Returns:
			List[ str ] | None: List of string values when available; otherwise None.
		"""
		return [ 'file_search_call.results',
		         'message.input_image.image_url',
		         'message.output_text.logprobs',
		         'reasoning.encrypted_content' ]
	
	@property
	def reasoning_options( self ) -> List[ str ] | None:
		"""Reasoning options.

		Purpose:
			Returns the reasoning-level names exposed by this wrapper for user-interface selectors
			and thinking configuration.

		Returns:
			List[ str ] | None: List of string values when available; otherwise None.
		"""
		return [ 'THINKING_LEVEL_UNSPECIFIED', 'MINIMAL',
		         'LOW', 'MEDIUM', 'HIGH' ]
	
	@property
	def choice_options( self ) -> List[ str ] | None:
		"""Choice options.

		Purpose:
			Returns tool-choice option names exposed by this wrapper for provider tool-selection
		         configuration.

		Returns:
			List[ str ] | None: List of string values when available; otherwise None.
		"""
		return [ 'AUTO',
		         'ANY',
		         'NONE',
		         'VALIDATED' ]
	
	@property
	def tool_options( self ) -> List[ str ] | None:
		"""Tool options.

		Purpose:
			Returns the tool names exposed by this wrapper for user-interface selectors and
		         provider request configuration.

		Returns:
			List[ str ] | None: List of string values when available; otherwise None.
		"""
		return [ 'google_search',
		         'google_maps',
		         'file_search',
		         'url_context',
		         'code_execution',
		         'computer_use' ]
	
	@property
	def modality_options( self ) -> List[ str ] | None:
		"""Modality options.

		Purpose:
			Returns response-modality option names exposed by this wrapper for text, image,
and audio request configuration.

		Returns:
			List[ str ] | None: List of string values when available; otherwise None.
		"""
		return [ 'MODALITY_UNSPECIFIED', 'TEXT', 'IMAGE', 'AUDIO' ]
	
	@property
	def media_options( self ) -> List[ str ] | None:
		"""Media options.

		Purpose:
			Returns media-resolution option names exposed by this wrapper for multimodal request
			configuration.

		Returns:
			List[ str ] | None: List of string values when available; otherwise None.
		"""
		return [ 'media_resolution_high',
		         'media_resolution_medium',
		         'media_resolution_low' ]
	
	def upload( self, filepath: str, name: str = None ) -> File | None:
		"""Upload a local resource to the provider.

		Purpose:
			Uploads a local file or blob to the provider-managed remote location represented by
			the active wrapper. The method validates required identifiers, stores request state,
			executes the upload, and returns provider metadata.

		Args:
			filepath: Local file path used by file upload or document workflows.
			name: Name value used by the active workflow, such as an argument name, display name,
			or remote object name.

		Returns:
			File | None: Provider file metadata when available; otherwise None.

		Raises:
			Error: Raised when validation, provider execution, storage access, or response
			processing fails.
		"""
		try:
			throw_if( 'filepath', filepath )
			throw_if( 'name', name )
			self.file_path = filepath;
			self.display_name = name
			self.client = genai.Client( api_key=self.gemini_api_key )
			self.response = self.client.files.upload( path=self.file_path,
				config={ 'display_name': self.display_name } )
			return self.response
		except Exception as e:
			ex = Error( e );
			ex.module = 'gemini'
			ex.cause = 'Files'
			ex.method = 'upload( self, path: str, name: str ) -> Optional[ File ]'
			Logger( ).write( ex )
			raise ex
	
	def list( self, model: str = 'gemini-2.0-flash', temperature: float = None,
			top_p: float = None, frequency: float = None, presence: float = None,
			max_tokens: int = None, stops: List[ str ] = None ) -> List[ str ]:
		"""List remote provider resources.

		Purpose:
			Lists files from the configured Google Cloud Storage location used by the file
			wrapper. The method stores request settings, queries the configured bucket and prefix,
			appends blob names to wrapper state, and returns the collected file names.

		Args:
			model: Gemini model identifier used for the provider request.
			temperature: Sampling temperature applied to generation when supplied.
			top_p: Nucleus sampling threshold applied to generation when supplied.
			frequency: Frequency penalty applied to repeated tokens when supplied.
			presence: Presence penalty applied to repeated topics or tokens when supplied.
			max_tokens: Maximum output token count supplied to provider configuration.
			stops: Stop sequences supplied to generation configuration.

		Returns:
			List[ str ]: List of string values produced by the workflow.

		Raises:
			Error: Raised when validation, provider execution, storage access, or response
			processing fails.
		"""
		try:
			self.model = model
			self.top_p = top_p;
			self.temperature = temperature
			self.frequency_penalty = frequency
			self.presence_penalty = presence
			self.max_tokens = max_tokens
			self.stops = stops
			self.storage_client = storage.Client( api_key=cfg.GOOGLE_API_KEY )
			name = "jeni-financial"
			prefix = "regulations"
			bucket = self.storage_client.bucket( bucket_name=name )
			for blob in bucket.list_blobs( prefix=prefix ):
				self.files.append( blob.name )
			return self.files
		except Exception as e:
			ex = Error( e );
			ex.module = 'gemini'
			ex.cause = 'Files'
			ex.method = 'list_files( self ) -> Optional[ List[ File ] ]'
			Logger( ).write( ex )
			raise ex
	
	def retrieve( self, file_id: str ) -> Optional[ File ]:
		"""Retrieve provider metadata for a remote resource.

		Purpose:
			Retrieves provider metadata for a remote file or blob. The method validates the
			identifier values, stores request state, executes the retrieval call, and returns the
			provider resource object when available.

		Args:
			file_id: Remote Gemini file identifier.

		Returns:
			Optional[ File ]: Provider file metadata when available; otherwise None.

		Raises:
			Error: Raised when validation, provider execution, storage access, or response
 processing fails.
		"""
		try:
			throw_if( 'file_id', file_id )
			self.file_id = file_id
			self.response = self.client.files.get( name=self.file_id )
			return self.response
		except Exception as e:
			ex = Error( e );
			ex.module = 'gemini'
			ex.cause = 'Files'
			ex.method = 'retrieve( self, file_id: str ) -> Optional[ File ]'
			Logger( ).write( ex )
			raise ex
	
	def summarize( self, prompt: str, filepath: str, model: str = 'gemini-2.0-flash',
			temperature: float = None, top_p: float = None, frequency: float = None,
			presence: float = None,
			max_tokens: int = None, stops: List[ str ] = None, instruct: str = None ) -> (
			str | None):
		"""Summarize.

		Purpose:
			Uploads a document and requests a Gemini-generated summary. The method validates
			prompt and file path values, prepares model configuration, supports Vertex and
			developer API file paths, and returns summary text.

		Args:
			prompt: User prompt, instruction, or query submitted to the provider workflow.
			filepath: Local file path used by file upload or document workflows.
			model: Gemini model identifier used for the provider request.
			temperature: Sampling temperature applied to generation when supplied.
			top_p: Nucleus sampling threshold applied to generation when supplied.
			frequency: Frequency penalty applied to repeated tokens when supplied.
			presence: Presence penalty applied to repeated topics or tokens when supplied.
			max_tokens: Maximum output token count supplied to provider configuration.
			stops: Stop sequences supplied to generation configuration.
			instruct: Optional system instruction supplied to provider configuration.

		Returns:
			str | None: Text output when available; otherwise None.

		Raises:
			Error: Raised when validation, provider execution, storage access, or response
			processing fails.
		"""
		try:
			throw_if( 'prompt', prompt )
			throw_if( 'filepath', filepath )
			self.prompt = prompt
			self.file_path = filepath
			self.model = model
			self.top_p = top_p;
			self.temperature = temperature
			self.frequency_penalty = frequency
			self.presence_penalty = presence
			self.max_tokens = max_tokens
			self.stops = stops
			self.instructions = instruct
			self.content_config = GenerateContentConfig( temperature=self.temperature )
			self.client = genai.Client( api_key=self.gemini_api_key )
			if self.use_vertex:
				with open( self.file_path, 'rb' ) as f:
					doc_part = Part.from_bytes( data=f.read( ), mime_type="application/pdf" )
				response = self.client.models.generate_content( model=self.model,
					contents=[ doc_part, self.prompt ], config=self.content_config )
			else:
				uploaded_file = self.client.files.upload( path=self.file_path )
				response = self.client.models.generate_content( model=self.model,
					contents=[ uploaded_file, self.prompt ], config=self.content_config )
			return response.text
		except Exception as e:
			ex = Error( e )
			ex.module = 'gemini'
			ex.cause = 'Files'
			ex.method = 'summarize_document( self, prompt, filepath, model ) -> str'
			Logger( ).write( ex )
			raise ex
	
	def search( self, prompt: str, filepath: str, model: str = 'gemini-2.0-flash',
			temperature: float = None, top_p: float = None, frequency: float = None,
			presence: float = None,
			max_tokens: int = None, stops: List[ str ] = None, instruct: str = None ) -> (
			str | None):
		"""Search.

		Purpose:
			Uploads a document and runs a Gemini document-search style request using the supplied
			prompt. The method validates prompt and file path values, prepares generation
			configuration, executes the provider call, and returns response text.

		Args:
			prompt: User prompt, instruction, or query submitted to the provider workflow.
			filepath: Local file path used by file upload or document workflows.
			model: Gemini model identifier used for the provider request.
			temperature: Sampling temperature applied to generation when supplied.
			top_p: Nucleus sampling threshold applied to generation when supplied.
			frequency: Frequency penalty applied to repeated tokens when supplied.
			presence: Presence penalty applied to repeated topics or tokens when supplied.
			max_tokens: Maximum output token count supplied to provider configuration.
			stops: Stop sequences supplied to generation configuration.
			instruct: Optional system instruction supplied to provider configuration.

		Returns:
			str | None: Text output when available; otherwise None.

		Raises:
			Error: Raised when validation, provider execution, storage access, or response
 processing fails.
		"""
		try:
			throw_if( 'prompt', prompt )
			throw_if( 'filepath', filepath )
			self.prompt = prompt
			self.file_path = filepath
			self.model = model
			self.top_p = top_p;
			self.temperature = temperature
			self.frequency_penalty = frequency
			self.presence_penalty = presence
			self.max_tokens = max_tokens
			self.stops = stops
			self.instructions = instruct
			self.content_config = GenerateContentConfig( temperature=self.temperature )
			self.client = genai.Client( api_key=self.gemini_api_key )
			if self.use_vertex:
				with open( self.file_path, 'rb' ) as f:
					doc_part = Part.from_bytes( data=f.read( ), mime_type="application/pdf" )
				response = self.client.models.generate_content( model=self.model,
					contents=[ doc_part, self.prompt ], config=self.content_config )
			else:
				uploaded_file = self.client.files.upload( path=self.file_path )
				response = self.client.models.generate_content( model=self.model,
					contents=[ uploaded_file,
					           self.prompt ], config=self.content_config )
			return response.text
		except Exception as e:
			ex = Error( e )
			ex.module = 'gemini'
			ex.cause = 'Files'
			ex.method = 'search( self, prompt, filepath, model ) -> str'
			Logger( ).write( ex )
			raise ex
	
	def survey( self, prompt: str, filepaths: List[ str ], model: str = 'gemini-2.0-flash',
			temperature: float = None, top_p: float = None, frequency: float = None,
			presence: float = None, max_tokens: int = None,
			stops: List[ str ] = None ) -> str | None:
		"""Survey.

		Purpose:
			Uploads multiple document paths and requests a Gemini response over the supplied
			collection. The method validates prompt and file path inputs, prepares generation
			configuration, and returns the provider response text.

		Args:
			prompt: User prompt, instruction, or query submitted to the provider workflow.
			filepaths: Collection of local file paths used by multi-document workflows.
			model: Gemini model identifier used for the provider request.
			temperature: Sampling temperature applied to generation when supplied.
			top_p: Nucleus sampling threshold applied to generation when supplied.
			frequency: Frequency penalty applied to repeated tokens when supplied.
			presence: Presence penalty applied to repeated topics or tokens when supplied.
			max_tokens: Maximum output token count supplied to provider configuration.
			stops: Stop sequences supplied to generation configuration.

		Returns:
			str | None: Text output when available; otherwise None.

		Raises:
			Error: Raised when validation, provider execution, storage access, or response
 processing fails.
		"""
		try:
			throw_if( 'prompt', prompt )
			throw_if( 'filepaths', filepaths )
			self.prompt = prompt
			self.file_paths = filepaths
			self.model = model
			self.top_p = top_p;
			self.temperature = temperature
			self.frequency_penalty = frequency
			self.presence_penalty = presence
			self.max_tokens = max_tokens
			self.stops = stops
			self.content_config = GenerateContentConfig( temperature=self.temperature )
			self.client = genai.Client( api_key=self.gemini_api_key )
			if self.use_vertex:
				with open( self.file_path, 'rb' ) as f:
					doc_part = Part.from_bytes( data=f.read( ), mime_type="application/pdf" )
				response = self.client.models.generate_content( model=self.model,
					contents=[ doc_part, self.prompt ], config=self.content_config )
			else:
				uploaded_file = self.client.files.upload( path=self.file_paths )
				response = self.client.models.generate_content( model=self.model,
					contents=[ uploaded_file, self.prompt ], config=self.content_config )
			return response.text
		except Exception as e:
			ex = Error( e )
			ex.module = 'gemini'
			ex.cause = 'Files'
			ex.method = 'survey( self, prompt, filepaths, model ) -> str'
			Logger( ).write( ex )
			raise ex
	
	def web_search( self, prompt: str, model: str = 'gemini-2.5-flash-lite',
			temperature: float = None,
			top_p: float = None, frequency: float = None, presence: float = None,
			max_tokens: int = None, stops: List[ str ] = None, instruct: str = None ) -> (
			str | None):
		"""Generate a search-grounded response.

		Purpose:
			Runs a search-grounded Gemini response from the file wrapper context. The method
			validates the prompt, builds a Google Search retrieval tool configuration, executes
			the provider request, and returns response text when available.

		Args:
			prompt: User prompt, instruction, or query submitted to the provider workflow.
			model: Gemini model identifier used for the provider request.
			temperature: Sampling temperature applied to generation when supplied.
			top_p: Nucleus sampling threshold applied to generation when supplied.
			frequency: Frequency penalty applied to repeated tokens when supplied.
			presence: Presence penalty applied to repeated topics or tokens when supplied.
			max_tokens: Maximum output token count supplied to provider configuration.
			stops: Stop sequences supplied to generation configuration.
			instruct: Optional system instruction supplied to provider configuration.

		Returns:
			str | None: Text output when available; otherwise None.
		"""
		try:
			throw_if( 'prompt', prompt )
			self.contents = prompt;
			self.model = model
			self.contents = prompt;
			self.top_p = top_p;
			self.temperature = temperature
			self.frequency_penalty = frequency
			self.presence_penalty = presence
			self.max_tokens = max_tokens
			self.stops = stops
			self.instructions = instruct
			self.tool_config = [
					types.Tool( google_search_retrieval=types.GoogleSearchRetrieval( ) ) ]
			self.content_config = GenerateContentConfig( temperature=self.temperature,
				tools=self.tool_config, system_instruction=self.instructions )
			self.client = genai.Client( api_key=self.gemini_api_key )
			response = self.client.models.generate_content( model=self.model,
				contents=self.contents, config=self.content_config )
			return response.text
		except Exception as e:
			exception = Error( e );
			exception.module = 'gemini'
			exception.cause = 'Chat'
			exception.method = 'web_search( self, prompt, model ) -> Optional[ str ]'
			Logger( ).write( exception )
	
	def search_maps( self, prompt: str, model: str = 'gemini-2.5-flash-lite',
			temperature: float = None, top_p: float = None, frequency: float = None,
			presence: float = None,
			max_tokens: int = None, stops: List[ str ] = None, instruct: str = None ) -> (
			str | None):
		"""Generate a location-grounded response.

		Purpose:
			Runs a location-oriented grounded Gemini response from the file wrapper context. The
			method validates the prompt, builds search retrieval configuration, executes the
			provider request, and returns response text when available.

		Args:
			prompt: User prompt, instruction, or query submitted to the provider workflow.
			model: Gemini model identifier used for the provider request.
			temperature: Sampling temperature applied to generation when supplied.
			top_p: Nucleus sampling threshold applied to generation when supplied.
			frequency: Frequency penalty applied to repeated tokens when supplied.
			presence: Presence penalty applied to repeated topics or tokens when supplied.
			max_tokens: Maximum output token count supplied to provider configuration.
			stops: Stop sequences supplied to generation configuration.
			instruct: Optional system instruction supplied to provider configuration.

		Returns:
			str | None: Text output when available; otherwise None.

		Raises:
			Error: Raised when validation, provider execution, storage access, or response
			processing fails.
		"""
		try:
			throw_if( 'prompt', prompt )
			self.contents = f"Using Google Search and Maps data, answer: {prompt}"
			self.model = model
			self.contents = prompt;
			self.top_p = top_p;
			self.temperature = temperature
			self.frequency_penalty = frequency
			self.presence_penalty = presence
			self.max_tokens = max_tokens
			self.stops = stops
			self.instructions = instruct
			self.tool_config = [
					types.Tool( google_search_retrieval=types.GoogleSearchRetrieval( ) ) ]
			self.content_config = GenerateContentConfig( temperature=self.temperature,
				tools=self.tool_config )
			self.client = genai.Client( api_key=self.gemini_api_key )
			response = self.client.models.generate_content( model=self.model,
				contents=self.contents, config=self.content_config )
			return response.text
		except Exception as e:
			exception = Error( e )
			exception.module = 'gemini'
			exception.cause = 'Chat'
			exception.method = 'search_maps( self, prompt, model ) -> Optional[ str ]'
			Logger( ).write( exception )
			raise exception
	
	def delete( self, file_id: str ) -> bool | None:
		"""Delete a remote provider resource.

		Purpose:
			Deletes the named Gemini file from remote file storage. The method validates the
			remote file identifier, creates a provider client, and executes the file delete
			request while preserving the existing return behavior.

		Args:
			file_id: Remote Gemini file identifier.

		Returns:
			bool | None: Boolean result when available; otherwise None.

		Raises:
			Error: Raised when validation, provider execution, storage access, or response
			processing fails.
		"""
		try:
			throw_if( 'file_id', file_id )
			self.file_id = file_id
			self.client = genai.Client( api_key=self.gemini_api_key )
			self.client.files.delete( name=self.file_id )
		except Exception as e:
			ex = Error( e );
			ex.module = 'gemini'
			ex.cause = 'FileStore'
			ex.method = 'delete( self, file_id: str ) -> bool'
			Logger( ).write( ex )
			raise ex

class VectorStores( Gemini ):
	"""Define VectorStores provider workflow state.

	Purpose:
		Wraps Google Cloud Storage operations used as a vector-store-style backend. The class
		treats buckets as collections and blobs as stored documents, providing create, upload,
		retrieve, list, search-grounding, map-search, and delete operations for remote assets.

	Attributes:
		project_id: Google Cloud project identifier used by storage workflows.
		bucket_name: Google Cloud Storage bucket name for the active operation.
		object_name: Google Cloud Storage object name for the active operation.
		file_path: Local file path used by file, image, audio, or document workflows.
		file_ids: Remote file identifiers tracked by the vector-store wrapper.
		store_ids: Remote store identifiers tracked by the vector-store wrapper.
		client: Provider client used to execute Gemini requests.
		bucket: Google Cloud Storage bucket object used by the active operation.
		response: Raw provider response captured by the active workflow.
		collections: Named storage collections exposed by the wrapper.
		documents: Named remote documents or blob identifiers tracked by the wrapper.
	"""
	project_id: Optional[ str ]
	bucket_name: Optional[ str ]
	object_name: Optional[ str ]
	file_path: Optional[ str ]
	file_ids: Optional[ List[ str ] ]
	store_ids: Optional[ List[ str ] ]
	client: Optional[ storage.Client ]
	bucket: Optional[ storage.Bucket ]
	response: Optional[ Any ]
	collections: Optional[ Dict[ str, str ] ]
	documents: Optional[ Dict[ str, str ] ]
	
	def __init__( self ) -> None:
		"""Initialize VectorStores runtime state.

		Purpose:
			Initializes VectorStores runtime attributes without executing provider requests. The
			constructor prepares configuration fields, client placeholders, request state,
response holders, and workflow-specific caches used by later methods.
		"""
		self.project_id = cfg.GOOGLE_CLOUD_PROJECT_ID
		self.client = storage.Client( project=self.project_id )
		self.bucket_name = None
		self.object_name = None
		self.file_path = None
		self.media_resolution = None
		self.file_ids = [ ]
		self.store_ids = [ ]
		self.stops = [ ]
		self.response_modalities = [ ]
		self.tools = [ ]
		self.domains = [ ]
		self.http_options = { }
		self.bucket = None
		self.response = None
		self.collections = \
			{
					'Federal Financial Data': 'jeni-financial/data',
					'Federal Financial Regulations': 'jeni-financial/regulations',
					'DoW Financial Data': 'jeni_dow/budget/data',
					'DoW Financial Regulations': 'jeni_dow/budget/regulations',
					'DoA Financial Data': 'jenni-doa/Financial Data',
			}
		self.documents = \
			{
					'Account_Balances.csv': 'file-U6wFeRGSeg38Db5uJzo5sj',
					'SF133.csv': 'file-32s641QK1Xb5QUatY3zfWF',
					'Authority.csv': 'file-Qi2rw2QsdxKBX1iiaQxY3m',
					'Outlays.csv': 'file-GHEwSWR7ezMvHrQ3X648wn'
			}
	
	@property
	def model_options( self ) -> List[ str ] | None:
		"""Model options.

		Purpose:
			Returns the model identifiers exposed by this wrapper for user-interface selectors and
			request configuration.

		Returns:
			List[ str ] | None: List of string values when available; otherwise None.
		"""
		return [ 'gemini-2.5-flash',
		         'gemini-2.5 flash image',
		         'gemini-2.5 flash-tts',
		         'gemini-2.5 flash-lite',
		         'gemini-2.0-flash',
		         'gemini-2.0-flash-lite' ]
	
	@property
	def media_options( self ) -> List[ str ] | None:
		"""Media options.

		Purpose:
			Returns media-resolution option names exposed by this wrapper for multimodal request
			configuration.

		Returns:
			List[ str ] | None: List of string values when available; otherwise None.
		"""
		return [ 'media_resolution_high',
		         'media_resolution_medium',
		         'media_resolution_low' ]
	
	def create( self, bucket: str, name: str ) -> bool:
		"""Create or modify the provider resource represented by the wrapper.

		Purpose:
			Deletes the named object from the supplied Google Cloud Storage bucket. The method
			validates bucket and object identifiers, stores the active request state, executes the
			blob delete operation, and returns True when the existing behavior completes
			successfully.

		Args:
			bucket: Google Cloud Storage bucket name.
			name: Name value used by the active workflow, such as an argument name, display name,
			or remote object name.

		Returns:
			bool: Boolean result produced by the workflow.

		Raises:
			Error: Raised when validation, provider execution, storage access, or response
			processing fails.
		"""
		try:
			throw_if( 'bucket', bucket )
			throw_if( 'name', name )
			self.bucket_name = bucket
			self.object_name = name
			self.bucket = self.client.bucket( self.bucket_name )
			blob = self.bucket.blob( self.object_name )
			blob.delete( )
			return True
		except Exception as e:
			ex = Error( e )
			ex.module = 'gemini'
			ex.cause = 'VectorStores'
			ex.method = 'delete( self, bucket, name )'
			Logger( ).write( ex )
			raise ex
	
	def upload( self, path: str, bucket: str, name: str = None ) -> Any:
		"""Upload a local resource to the provider.

		Purpose:
			Uploads a local file or blob to the provider-managed remote location represented by
			the active wrapper. The method validates required identifiers, stores request state,
			executes the upload, and returns provider metadata.

		Args:
			path: Local file path used by image, audio, or cloud-storage workflows.
			bucket: Google Cloud Storage bucket name.
			name: Name value used by the active workflow, such as an argument name, display name,
or remote object name.

		Returns:
			Any: Provider-specific object produced by the workflow.

		Raises:
			Error: Raised when validation, provider execution, storage access, or response
			processing fails.
		"""
		try:
			throw_if( 'path', path )
			throw_if( 'bucket', bucket )
			self.file_path = path
			self.bucket_name = bucket
			self.object_name = name or path.split( '/' )[ -1 ]
			self.bucket = self.client.bucket( self.bucket_name )
			blob = self.bucket.blob( self.object_name )
			blob.upload_from_filename( self.file_path )
			self.response = blob
			return blob
		except Exception as e:
			ex = Error( e )
			ex.module = 'gemini'
			ex.cause = 'VectorStores'
			ex.method = 'upload( self, path, bucket, name )'
			Logger( ).write( ex )
			raise ex
	
	def retrieve( self, bucket: str, name: str ) -> Any:
		"""Retrieve provider metadata for a remote resource.

		Purpose:
			Retrieves provider metadata for a remote file or blob. The method validates the
			identifier values, stores request state, executes the retrieval call, and returns the
			provider resource object when available.

		Args:
			bucket: Google Cloud Storage bucket name.
			name: Name value used by the active workflow, such as an argument name, display name,
			or remote object name.

		Returns:
			Any: Provider-specific object produced by the workflow.

		Raises:
			Error: Raised when validation, provider execution, storage access, or response
			processing fails.
		"""
		try:
			throw_if( 'bucket', bucket )
			throw_if( 'name', name )
			self.bucket_name = bucket
			self.object_name = name
			self.bucket = self.client.bucket( self.bucket_name )
			blob = self.bucket.get_blob( self.object_name )
			return blob
		except Exception as e:
			ex = Error( e )
			ex.module = 'gemini'
			ex.cause = 'VectorStores'
			ex.method = 'retrieve( self, bucket, name )'
			Logger( ).write( ex )
			raise ex
	
	def list( self, bucket: str ) -> List[ Any ]:
		"""List remote provider resources.

		Purpose:
			Lists remote provider resources represented by the active wrapper. The method stores
			query state, retrieves available files or blobs, updates local tracking collections
			when applicable, and returns the retrieved resource list.

		Args:
			bucket: Google Cloud Storage bucket name.

		Returns:
			List[ Any ]: Provider-specific objects produced by the workflow.

		Raises:
			Error: Raised when validation, provider execution, storage access, or response
			processing fails.
		"""
		try:
			throw_if( 'bucket', bucket )
			self.bucket_name = bucket
			self.bucket = self.client.bucket( self.bucket_name )
			blobs = list( self.bucket.list_blobs( ) )
			self.documents = { blob.name: blob.id for blob in blobs }
			return blobs
		except Exception as e:
			ex = Error( e )
			ex.module = 'gemini'
			ex.cause = 'VectorStores'
			ex.method = 'list( self, bucket )'
			Logger( ).write( ex )
			raise ex
	
	def web_search( self, prompt: str, model: str = 'gemini-2.5-flash-lite',
			temperature: float = None, top_p: float = None, frequency: float = None,
			presence: float = None,
			max_tokens: int = None, stops: List[ str ] = None, instruct: str = None ) -> (
			str | None):
		"""Generate a search-grounded response.

		Purpose:
			Runs a search-grounded Gemini response from the vector-store wrapper context. The
			method validates the prompt, builds a Google Search retrieval tool configuration,
			executes the provider request, and returns response text when available.

		Args:
			prompt: User prompt, instruction, or query submitted to the provider workflow.
			model: Gemini model identifier used for the provider request.
			temperature: Sampling temperature applied to generation when supplied.
			top_p: Nucleus sampling threshold applied to generation when supplied.
			frequency: Frequency penalty applied to repeated tokens when supplied.
			presence: Presence penalty applied to repeated topics or tokens when supplied.
			max_tokens: Maximum output token count supplied to provider configuration.
			stops: Stop sequences supplied to generation configuration.
			instruct: Optional system instruction supplied to provider configuration.

		Returns:
			str | None: Text output when available; otherwise None.

		Raises:
			Error: Raised when validation, provider execution, storage access, or response
			processing fails.
		"""
		try:
			throw_if( 'prompt', prompt )
			self.contents = prompt;
			self.model = model
			self.contents = prompt;
			self.top_p = top_p;
			self.temperature = temperature
			self.frequency_penalty = frequency
			self.presence_penalty = presence
			self.max_tokens = max_tokens
			self.stops = stops
			self.instructions = instruct
			self.tool_config = [
					types.Tool( google_search_retrieval=types.GoogleSearchRetrieval( ) ) ]
			self.content_config = GenerateContentConfig( temperature=self.temperature,
				tools=self.tool_config, system_instruction=self.instructions )
			self.client = genai.Client( api_key=self.gemini_api_key )
			response = self.client.models.generate_content( model=self.model,
				contents=self.contents, config=self.content_config )
			return response.text
		except Exception as e:
			exception = Error( e );
			exception.module = 'gemini'
			exception.cause = 'Chat'
			exception.method = 'web_search( self, prompt, model ) -> Optional[ str ]'
			Logger( ).write( exception )
			raise exception
	
	def search_maps( self, prompt: str, model: str = 'gemini-2.5-flash-lite',
			temperature: float = None,
			top_p: float = None, frequency: float = None, presence: float = None,
			max_tokens: int = None, stops: List[ str ] = None, instruct: str = None ) -> (
			str | None):
		"""Generate a location-grounded response.

		Purpose:
			Runs a location-oriented grounded Gemini response from the vector-store wrapper
			context. The method validates the prompt, builds search retrieval configuration,
			executes the provider request, and returns response text when available.

		Args:
			prompt: User prompt, instruction, or query submitted to the provider workflow.
			model: Gemini model identifier used for the provider request.
			temperature: Sampling temperature applied to generation when supplied.
			top_p: Nucleus sampling threshold applied to generation when supplied.
			frequency: Frequency penalty applied to repeated tokens when supplied.
			presence: Presence penalty applied to repeated topics or tokens when supplied.
			max_tokens: Maximum output token count supplied to provider configuration.
			stops: Stop sequences supplied to generation configuration.
			instruct: Optional system instruction supplied to provider configuration.

		Returns:
			str | None: Text output when available; otherwise None.

		Raises:
			Error: Raised when validation, provider execution, storage access, or response
			processing fails.
		"""
		try:
			throw_if( 'prompt', prompt )
			self.contents = f"Using Google Search and Maps data, answer: {prompt}"
			self.model = model
			self.contents = prompt;
			self.top_p = top_p;
			self.temperature = temperature
			self.frequency_penalty = frequency
			self.presence_penalty = presence
			self.max_tokens = max_tokens
			self.stops = stops
			self.instructions = instruct
			self.tool_config = [
					types.Tool( google_search_retrieval=types.GoogleSearchRetrieval( ) ) ]
			self.content_config = GenerateContentConfig( temperature=self.temperature,
				tools=self.tool_config )
			self.client = genai.Client( api_key=self.gemini_api_key )
			response = self.client.models.generate_content( model=self.model,
				contents=self.contents, config=self.content_config )
			return response.text
		except Exception as e:
			exception = Error( e )
			exception.module = 'gemini'
			exception.cause = 'Chat'
			exception.method = 'search_maps( self, prompt, model ) -> Optional[ str ]'
			Logger( ).write( exception )
			raise exception
	
	def delete( self, bucket: str, name: str ) -> bool:
		"""Delete a remote provider resource.

		Purpose:
			Deletes the named object from the supplied Google Cloud Storage bucket. The method
			validates bucket and object identifiers, stores the active request state, executes the
			blob delete operation, and returns True when deletion completes.

		Args:
			bucket: Google Cloud Storage bucket name.
			name: Name value used by the active workflow, such as an argument name, display name,
			or remote object name.

		Returns:
			bool: Boolean result produced by the workflow.

		Raises:
			Error: Raised when validation, provider execution, storage access, or response
			processing fails.
		"""
		try:
			throw_if( 'bucket', bucket )
			throw_if( 'name', name )
			self.bucket_name = bucket
			self.object_name = name
			self.bucket = self.client.bucket( self.bucket_name )
			blob = self.bucket.blob( self.object_name )
			blob.delete( )
			return True
		except Exception as e:
			ex = Error( e )
			ex.module = 'gemini'
			ex.cause = 'VectorStores'
			ex.method = 'delete( self, bucket, name )'
			Logger( ).write( ex )
			raise ex
