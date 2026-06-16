/*
 * ==========================================================================================
 *  Jimi Documentation JavaScript
 *  File: docs/assets/js/jimi.js
 *
 *  Purpose:
 *      Provides progressive enhancements for the Jimi MkDocs Material documentation site.
 *      The script improves API navigation, table filtering, code-block readability, heading
 *      linking, page tools, scroll position behavior, and reading progress without requiring
 *      external libraries.
 *
 *  Compatibility:
 *      - MkDocs Material
 *      - mkdocstrings
 *      - Modern Chromium, Edge, Firefox, Safari
 *
 *  Notes:
 *      This script avoids network calls, analytics, cookies, and storage of user content.
 * ==========================================================================================
 */
( function()
{
	"use strict";
	const JimiDocs = {
		config: {
			initializedAttribute: "data-jimi-enhanced",
			progressId: "jimi-reading-progress",
			scrollTopId: "jimi-scroll-top",
			pageToolsId: "jimi-page-tools",
			navScrollKey: "jimi-docs-nav-scroll",
			contentSelector: ".md-content__inner",
			headingSelector: ".md-typeset h2[id], .md-typeset h3[id], .md-typeset h4[id]",
			tableSelector: ".md-typeset table:not([data-jimi-no-filter])",
			codeSelector: ".md-typeset pre > code",
			navSelector: ".md-nav--primary .md-nav__list",
			tocSelector: ".md-nav--secondary",
			apiObjectSelector:
					".doc.doc-object, .doc-class, .doc-function, .doc-method, .doc-attribute, .doc-property",
			headingLinkClass: "jimi-heading-link",
			tableFilterClass: "jimi-table-filter",
			codeLabelClass: "jimi-code-label",
			codeToggleClass: "jimi-code-toggle",
			codeCollapsedClass: "jimi-code-collapsed",
			apiHiddenClass: "jimi-api-hidden",
			tocActiveClass: "jimi-toc-active",
			largeTableMinimumRows: 8,
			maxCollapsedCodeHeight: 420
		},
		state: {
			scrollTicking: false,
			resizeTicking: false
		},
		init: function()
		{
			if( document.documentElement.getAttribute( this.config.initializedAttribute ) ===
					"true" )
			{
				return;
			}
			document.documentElement.setAttribute( this.config.initializedAttribute, "true" );
			this.enhanceExternalLinks();
			this.customizeSearch();
			this.addReadingProgress();
			this.addScrollTopButton();
			this.addPageTools();
			this.addHeadingLinks();
			this.addTableFilters();
			this.addCodeLabels();
			this.addCodeToggles();
			this.addPagePathMetadata();
			this.restoreNavigationScroll();
			this.enhanceKeyboardFocus();
			this.enhanceApiReference();
			this.addApiTools();
			this.updateReadingProgress();
			this.updateScrollTopVisibility();
			this.updateTocProgress();
			this.bindLifecycleEvents();
		},
		bindLifecycleEvents: function()
		{
			const self = this;
			window.addEventListener( "scroll", function()
			{
				if( !self.state.scrollTicking )
				{
					window.requestAnimationFrame( function()
					{
						self.updateReadingProgress();
						self.updateScrollTopVisibility();
						self.updateTocProgress();
						self.state.scrollTicking = false;
					} );
					self.state.scrollTicking = true;
				}
			}, { passive: true } );
			window.addEventListener( "resize", function()
			{
				if( !self.state.resizeTicking )
				{
					window.requestAnimationFrame( function()
					{
						self.updateReadingProgress();
						self.updateTocProgress();
						self.state.resizeTicking = false;
					} );
					self.state.resizeTicking = true;
				}
			}, { passive: true } );
			document.addEventListener( "click", function( event )
			{
				self.handleDocumentClick( event );
			} );
			document.addEventListener( "keydown", function( event )
			{
				self.handleKeyboardShortcuts( event );
			} );
			window.addEventListener( "beforeunload", function()
			{
				self.saveNavigationScroll();
			} );
			if( typeof document$ !== "undefined" && document$ && typeof document$.subscribe ===
					"function" )
			{
				document$.subscribe( function()
				{
					document.documentElement.removeAttribute( self.config.initializedAttribute );
					setTimeout( function()
					{
						self.init();
					}, 25 );
				} );
			}
		},
		handleDocumentClick: function( event )
		{
			const target = event.target;
			if( !target )
			{
				return;
			}
			if( target.closest && target.closest( "#" + this.config.scrollTopId ) )
			{
				event.preventDefault();
				this.scrollToTop();
				return;
			}
			if( target.closest && target.closest( "[data-jimi-copy-heading]" ) )
			{
				event.preventDefault();
				this.copyHeadingLink( target.closest( "[data-jimi-copy-heading]" ) );
				return;
			}
			if( target.closest && target.closest( "[data-jimi-copy-page]" ) )
			{
				event.preventDefault();
				this.copyPageLink( target.closest( "[data-jimi-copy-page]" ) );
				return;
			}
			if( target.closest && target.closest( "[data-jimi-print-page]" ) )
			{
				event.preventDefault();
				window.print();
				return;
			}
			if( target.closest && target.closest( "[data-jimi-toggle-code]" ) )
			{
				event.preventDefault();
				this.toggleCodeBlock( target.closest( "[data-jimi-toggle-code]" ) );
				return;
			}
			if( target.closest && target.closest( "[data-jimi-api-expand]" ) )
			{
				event.preventDefault();
				this.setApiDetailsState( true );
				return;
			}
			if( target.closest && target.closest( "[data-jimi-api-collapse]" ) )
			{
				event.preventDefault();
				this.setApiDetailsState( false );
				return;
			}
			if( target.closest && target.closest( "[data-jimi-api-clear]" ) )
			{
				event.preventDefault();
				this.clearApiFilter();
			}
		},
		handleKeyboardShortcuts: function( event )
		{
			const key = ( event.key || "" ).toLowerCase();
			if( event.altKey && key === "t" )
			{
				event.preventDefault();
				this.scrollToTop();
			}
			if( event.altKey && key === "p" )
			{
				event.preventDefault();
				window.print();
			}
			if( event.altKey && key === "l" )
			{
				event.preventDefault();
				this.copyCurrentPageToClipboard();
			}
			if( event.altKey && key === "f" )
			{
				const apiSearch = document.getElementById( "jimi-api-search" );
				if( apiSearch )
				{
					event.preventDefault();
					apiSearch.focus();
				}
			}
		},
		enhanceExternalLinks: function()
		{
			const links = document.querySelectorAll( ".md-typeset a[href]" );
			const currentHost = window.location.host;
			links.forEach( function( link )
			{
				try
				{
					const url = new URL( link.href, window.location.href );
					if( url.host && url.host !== currentHost )
					{
						link.setAttribute( "target", "_blank" );
						link.setAttribute( "rel", "noopener noreferrer" );
						link.classList.add( "jimi-external-link" );
						if( !link.querySelector( ".jimi-external-indicator" ) )
						{
							const indicator = document.createElement( "span" );
							indicator.className = "jimi-external-indicator";
							indicator.setAttribute( "aria-hidden", "true" );
							indicator.textContent = " ↗";
							link.appendChild( indicator );
						}
					}
				}
				catch( error )
				{
					return;
				}
			} );
		},
		customizeSearch: function()
		{
			const searchInputs = document.querySelectorAll( "input.md-search__input" );
			searchInputs.forEach( function( input )
			{
				input.setAttribute( "placeholder", "Search Jimi docs..." );
				input.setAttribute( "aria-label", "Search Jimi documentation" );
			} );
		},
		addReadingProgress: function()
		{
			if( document.getElementById( this.config.progressId ) )
			{
				return;
			}
			const progress = document.createElement( "div" );
			progress.id = this.config.progressId;
			progress.setAttribute( "aria-hidden", "true" );
			progress.innerHTML = "<span></span>";
			document.body.appendChild( progress );
		},
		updateReadingProgress: function()
		{
			const progress = document.querySelector( "#" + this.config.progressId + " span" );
			if( !progress )
			{
				return;
			}
			const content = document.querySelector( this.config.contentSelector );
			const scrollTop = window.scrollY || document.documentElement.scrollTop;
			if( content )
			{
				const rect = content.getBoundingClientRect();
				const contentTop = rect.top + scrollTop;
				const contentHeight = Math.max( content.offsetHeight, 1 );
				const contentScroll = Math.min( Math.max( scrollTop - contentTop, 0 ),
						contentHeight );
				const percent = Math.min( Math.max( contentScroll / contentHeight, 0 ), 1 );
				progress.style.width = ( percent * 100 ).toFixed( 2 ) + "%";
				return;
			}
			let maxScroll = document.documentElement.scrollHeight - window.innerHeight;
			if( maxScroll <= 0 )
			{
				maxScroll = 1;
			}
			progress.style.width =
					Math.min( Math.max( ( scrollTop / maxScroll ) * 100, 0 ), 100 ).toFixed( 2 ) +
					"%";
		},
		addScrollTopButton: function()
		{
			if( document.getElementById( this.config.scrollTopId ) )
			{
				return;
			}
			const button = document.createElement( "button" );
			button.id = this.config.scrollTopId;
			button.type = "button";
			button.className = "jimi-scroll-top";
			button.setAttribute( "aria-label", "Scroll to top" );
			button.setAttribute( "title", "Scroll to top (Alt+T)" );
			button.innerHTML = "↑";
			document.body.appendChild( button );
		},
		updateScrollTopVisibility: function()
		{
			const button = document.getElementById( this.config.scrollTopId );
			if( !button )
			{
				return;
			}
			if( ( window.scrollY || document.documentElement.scrollTop ) > 420 )
			{
				button.classList.add( "is-visible" );
			}
			else
			{
				button.classList.remove( "is-visible" );
			}
		},
		scrollToTop: function()
		{
			window.scrollTo( {
				top: 0,
				behavior: "smooth"
			} );
		},
		addPageTools: function()
		{
			if( document.getElementById( this.config.pageToolsId ) )
			{
				return;
			}
			const content = document.querySelector( this.config.contentSelector );
			if( !content )
			{
				return;
			}
			const title = content.querySelector( "h1" );
			if( !title )
			{
				return;
			}
			const tools = document.createElement( "div" );
			tools.id = this.config.pageToolsId;
			tools.className = "jimi-page-tools";
			tools.innerHTML = [
				"<button type=\"button\" data-jimi-copy-page title=\"Copy page link\" aria-label=\"Copy page link\">Copy link</button>",
				"<button type=\"button\" data-jimi-print-page title=\"Print page\" aria-label=\"Print page\">Print</button>"
			].join( "" );
			title.insertAdjacentElement( "afterend", tools );
		},
		copyPageLink: function( button )
		{
			this.copyTextToClipboard( window.location.href, button, "Copied", "Copy link" );
		},
		copyCurrentPageToClipboard: function()
		{
			const button = document.querySelector( "[data-jimi-copy-page]" );
			this.copyTextToClipboard( window.location.href, button, "Copied", "Copy link" );
		},
		addHeadingLinks: function()
		{
			const headings = document.querySelectorAll( this.config.headingSelector );
			headings.forEach( function( heading )
			{
				if( heading.querySelector( "." + JimiDocs.config.headingLinkClass ) )
				{
					return;
				}
				const button = document.createElement( "button" );
				button.type = "button";
				button.className = JimiDocs.config.headingLinkClass;
				button.setAttribute( "data-jimi-copy-heading", heading.id );
				button.setAttribute( "aria-label", "Copy link to " + heading.textContent.trim() );
				button.setAttribute( "title", "Copy section link" );
				button.textContent = "§";
				heading.appendChild( button );
			} );
		},
		copyHeadingLink: function( button )
		{
			const id = button.getAttribute( "data-jimi-copy-heading" );
			if( !id )
			{
				return;
			}
			const url = window.location.origin +
					window.location.pathname +
					window.location.search +
					"#" +
					encodeURIComponent( id );
			this.copyTextToClipboard( url, button, "Copied", "§" );
		},
		copyTextToClipboard: function( text, button, successText, defaultText )
		{
			const updateButton = function()
			{
				if( !button )
				{
					return;
				}
				const previousText = button.textContent;
				button.textContent = successText || "Copied";
				setTimeout( function()
				{
					button.textContent = defaultText || previousText;
				}, 1400 );
			};
			if( navigator.clipboard && typeof navigator.clipboard.writeText === "function" )
			{
				navigator.clipboard.writeText( text ).then( updateButton ).catch( function()
				{
					JimiDocs.fallbackCopyText( text );
					updateButton();
				} );
				return;
			}
			this.fallbackCopyText( text );
			updateButton();
		},
		fallbackCopyText: function( text )
		{
			const textarea = document.createElement( "textarea" );
			textarea.value = text;
			textarea.setAttribute( "readonly", "readonly" );
			textarea.style.position = "fixed";
			textarea.style.top = "-9999px";
			textarea.style.left = "-9999px";
			document.body.appendChild( textarea );
			textarea.select();
			try
			{
				document.execCommand( "copy" );
			}
			catch( error )
			{
				return;
			}
			finally
			{
				document.body.removeChild( textarea );
			}
		},
		addTableFilters: function()
		{
			const tables = document.querySelectorAll( this.config.tableSelector );
			tables.forEach( function( table, index )
			{
				if( table.getAttribute( "data-jimi-filtered" ) === "true" )
				{
					return;
				}
				const tbody = table.querySelector( "tbody" );
				if( !tbody )
				{
					return;
				}
				const rows = Array.prototype.slice.call( tbody.querySelectorAll( "tr" ) );
				if( rows.length < JimiDocs.config.largeTableMinimumRows )
				{
					return;
				}
				table.setAttribute( "data-jimi-filtered", "true" );
				const wrapper = document.createElement( "div" );
				wrapper.className = "jimi-table-tools";
				const input = document.createElement( "input" );
				input.type = "search";
				input.className = JimiDocs.config.tableFilterClass;
				input.placeholder = "Filter table...";
				input.setAttribute( "aria-label", "Filter table " + ( index + 1 ) );
				const count = document.createElement( "span" );
				count.className = "jimi-table-count";
				count.textContent = rows.length + " rows";
				wrapper.appendChild( input );
				wrapper.appendChild( count );
				table.parentNode.insertBefore( wrapper, table );
				input.addEventListener( "input", function()
				{
					JimiDocs.filterTable( table, input.value, count );
				} );
			} );
		},
		filterTable: function( table, query, countElement )
		{
			const normalizedQuery = ( query || "" ).toLowerCase().trim();
			const rows = Array.prototype.slice.call( table.querySelectorAll( "tbody tr" ) );
			let visible = 0;
			rows.forEach( function( row )
			{
				const text = row.textContent.toLowerCase();
				if( !normalizedQuery || text.indexOf( normalizedQuery ) !== -1 )
				{
					row.style.display = "";
					visible += 1;
				}
				else
				{
					row.style.display = "none";
				}
			} );
			if( countElement )
			{
				countElement.textContent = visible + " / " + rows.length + " rows";
			}
		},
		addCodeLabels: function()
		{
			const codeBlocks = document.querySelectorAll( this.config.codeSelector );
			codeBlocks.forEach( function( code )
			{
				const pre = code.parentElement;
				if( !pre || pre.getAttribute( "data-jimi-labeled" ) === "true" )
				{
					return;
				}
				const language = JimiDocs.detectCodeLanguage( code );
				if( !language )
				{
					return;
				}
				pre.setAttribute( "data-jimi-labeled", "true" );
				const label = document.createElement( "div" );
				label.className = JimiDocs.config.codeLabelClass;
				label.textContent = language;
				pre.insertAdjacentElement( "beforebegin", label );
			} );
		},
		detectCodeLanguage: function( code )
		{
			const className = code.className || "";
			const match = className.match( /language-([a-zA-Z0-9_+-]+)/ );
			if( match && match[ 1 ] )
			{
				return this.formatLanguageName( match[ 1 ] );
			}
			return "";
		},
		formatLanguageName: function( language )
		{
			const normalized = ( language || "" ).toLowerCase();
			const names = {
				py: "Python",
				python: "Python",
				ps1: "PowerShell",
				powershell: "PowerShell",
				bash: "Bash",
				sh: "Shell",
				shell: "Shell",
				yaml: "YAML",
				yml: "YAML",
				json: "JSON",
				js: "JavaScript",
				javascript: "JavaScript",
				css: "CSS",
				html: "HTML",
				markdown: "Markdown",
				md: "Markdown",
				text: "Text",
				sql: "SQL"
			};
			return names[ normalized ] || language.toUpperCase();
		},
		addCodeToggles: function()
		{
			const codeBlocks = document.querySelectorAll( ".md-typeset pre" );
			codeBlocks.forEach( function( pre, index )
			{
				if( pre.getAttribute( "data-jimi-toggle-ready" ) === "true" )
				{
					return;
				}
				if( pre.scrollHeight <= JimiDocs.config.maxCollapsedCodeHeight )
				{
					return;
				}
				pre.setAttribute( "data-jimi-toggle-ready", "true" );
				pre.classList.add( JimiDocs.config.codeCollapsedClass );
				pre.style.maxHeight = JimiDocs.config.maxCollapsedCodeHeight + "px";
				const button = document.createElement( "button" );
				button.type = "button";
				button.className = JimiDocs.config.codeToggleClass;
				button.setAttribute( "data-jimi-toggle-code", String( index ) );
				button.setAttribute( "aria-expanded", "false" );
				button.textContent = "Expand code";
				pre.insertAdjacentElement( "afterend", button );
			} );
		},
		toggleCodeBlock: function( button )
		{
			if( !button )
			{
				return;
			}
			const pre = button.previousElementSibling;
			if( !pre || pre.tagName.toLowerCase() !== "pre" )
			{
				return;
			}
			const expanded = button.getAttribute( "aria-expanded" ) === "true";
			if( expanded )
			{
				pre.classList.add( this.config.codeCollapsedClass );
				pre.style.maxHeight = this.config.maxCollapsedCodeHeight + "px";
				button.setAttribute( "aria-expanded", "false" );
				button.textContent = "Expand code";
			}
			else
			{
				pre.classList.remove( this.config.codeCollapsedClass );
				pre.style.maxHeight = "none";
				button.setAttribute( "aria-expanded", "true" );
				button.textContent = "Collapse code";
			}
		},
		addPagePathMetadata: function()
		{
			const content = document.querySelector( this.config.contentSelector );
			if( !content || content.querySelector( ".jimi-page-path" ) )
			{
				return;
			}
			const title = content.querySelector( "h1" );
			if( !title )
			{
				return;
			}
			const path = document.createElement( "div" );
			path.className = "jimi-page-path";
			path.setAttribute( "aria-label", "Current page path" );
			path.textContent = window.location.pathname.replace( /\/$/, "" ) || "/";
			title.insertAdjacentElement( "afterend", path );
		},
		restoreNavigationScroll: function()
		{
			const nav = document.querySelector( this.config.navSelector );
			if( !nav )
			{
				return;
			}
			try
			{
				const saved = sessionStorage.getItem( this.config.navScrollKey );
				if( saved !== null )
				{
					nav.scrollTop = parseInt( saved, 10 ) || 0;
				}
			}
			catch( error )
			{
				return;
			}
		},
		saveNavigationScroll: function()
		{
			const nav = document.querySelector( this.config.navSelector );
			if( !nav )
			{
				return;
			}
			try
			{
				sessionStorage.setItem( this.config.navScrollKey, String( nav.scrollTop || 0 ) );
			}
			catch( error )
			{
				return;
			}
		},
		enhanceKeyboardFocus: function()
		{
			document.addEventListener( "keydown", function( event )
			{
				if( event.key === "Tab" )
				{
					document.documentElement.classList.add( "jimi-keyboard-focus" );
				}
			} );
			document.addEventListener( "mousedown", function()
			{
				document.documentElement.classList.remove( "jimi-keyboard-focus" );
			} );
		},
		enhanceApiReference: function()
		{
			const objects = document.querySelectorAll( this.config.apiObjectSelector );
			objects.forEach( function( object )
			{
				if( object.getAttribute( "data-jimi-api-enhanced" ) === "true" )
				{
					return;
				}
				object.setAttribute( "data-jimi-api-enhanced", "true" );
				const heading = object.querySelector( ".doc-heading, h2, h3, h4" );
				if( !heading )
				{
					return;
				}
				const type = JimiDocs.detectApiObjectType( object );
				if( !type )
				{
					return;
				}
				const badge = document.createElement( "span" );
				badge.className = "jimi-badge";
				badge.textContent = type;
				heading.appendChild( document.createTextNode( " " ) );
				heading.appendChild( badge );
			} );
		},
		detectApiObjectType: function( object )
		{
			const className = object.className || "";
			if( className.indexOf( "doc-class" ) !== -1 )
			{
				return "class";
			}
			if( className.indexOf( "doc-method" ) !== -1 )
			{
				return "method";
			}
			if( className.indexOf( "doc-function" ) !== -1 )
			{
				return "function";
			}
			if( className.indexOf( "doc-attribute" ) !== -1 )
			{
				return "attribute";
			}
			if( className.indexOf( "doc-property" ) !== -1 )
			{
				return "property";
			}
			return "";
		},
		addApiTools: function()
		{
			const content = document.querySelector( this.config.contentSelector );
			if( !content || document.getElementById( "jimi-api-tools" ) )
			{
				return;
			}
			const apiObjects = content.querySelectorAll( this.config.apiObjectSelector );
			if( !apiObjects || apiObjects.length === 0 )
			{
				return;
			}
			const title = content.querySelector( "h1" );
			if( !title )
			{
				return;
			}
			const panel = document.createElement( "div" );
			panel.id = "jimi-api-tools";
			panel.className = "jimi-api-tools";
			panel.innerHTML = [
				"<h2 class=\"jimi-api-tools-title\">API Tools</h2>",
				"<label class=\"jimi-api-search-label\" for=\"jimi-api-search\">Filter API reference</label>",
				"<input id=\"jimi-api-search\" class=\"jimi-api-search\" type=\"search\" placeholder=\"Filter classes, methods, functions, and attributes...\" />",
				"<div class=\"jimi-api-tool-buttons\">",
				"<button class=\"jimi-api-tool-button\" type=\"button\" data-jimi-api-expand>Expand all</button>",
				"<button class=\"jimi-api-tool-button\" type=\"button\" data-jimi-api-collapse>Collapse all</button>",
				"<button class=\"jimi-api-tool-button\" type=\"button\" data-jimi-api-clear>Clear filter</button>",
				"</div>",
				"<p class=\"jimi-api-filter-status\" id=\"jimi-api-filter-status\">" +
				apiObjects.length + " API objects</p>"
			].join( "" );
			title.insertAdjacentElement( "afterend", panel );
			const input = document.getElementById( "jimi-api-search" );
			if( input )
			{
				input.addEventListener( "input", function()
				{
					JimiDocs.filterApiObjects( input.value );
				} );
			}
		},
		filterApiObjects: function( query )
		{
			const normalizedQuery = ( query || "" ).toLowerCase().trim();
			const objects = document.querySelectorAll( this.config.apiObjectSelector );
			const status = document.getElementById( "jimi-api-filter-status" );
			let visible = 0;
			objects.forEach( function( object )
			{
				const text = object.textContent.toLowerCase();
				if( !normalizedQuery || text.indexOf( normalizedQuery ) !== -1 )
				{
					object.classList.remove( JimiDocs.config.apiHiddenClass );
					visible += 1;
				}
				else
				{
					object.classList.add( JimiDocs.config.apiHiddenClass );
				}
			} );
			if( status )
			{
				status.textContent = visible + " / " + objects.length + " API objects";
			}
		},
		clearApiFilter: function()
		{
			const input = document.getElementById( "jimi-api-search" );
			if( input )
			{
				input.value = "";
			}
			this.filterApiObjects( "" );
		},
		setApiDetailsState: function( open )
		{
			const details = document.querySelectorAll( ".md-typeset details" );
			details.forEach( function( detail )
			{
				detail.open = Boolean( open );
			} );
		},
		updateTocProgress: function()
		{
			const toc = document.querySelector( this.config.tocSelector );
			const headings = Array.prototype.slice.call(
					document.querySelectorAll( this.config.headingSelector ) );
			if( !toc || headings.length === 0 )
			{
				return;
			}
			let activeId = "";
			for( let i = 0; i < headings.length; i += 1 )
			{
				const rect = headings[ i ].getBoundingClientRect();
				if( rect.top <= 120 )
				{
					activeId = headings[ i ].id;
				}
			}
			if( !activeId && headings[ 0 ] )
			{
				activeId = headings[ 0 ].id;
			}
			const links = toc.querySelectorAll( "a[href^='#']" );
			links.forEach( function( link )
			{
				const href = link.getAttribute( "href" ) || "";
				const id = decodeURIComponent( href.replace( /^#/, "" ) );
				if( id === activeId )
				{
					link.classList.add( JimiDocs.config.tocActiveClass );
				}
				else
				{
					link.classList.remove( JimiDocs.config.tocActiveClass );
				}
			} );
		}
	};
	
	function startJimiDocs()
	{
		try
		{
			JimiDocs.init();
		}
		catch( error )
		{
			return;
		}
	}
	
	if( document.readyState === "loading" )
	{
		document.addEventListener( "DOMContentLoaded", startJimiDocs );
	}
	else
	{
		startJimiDocs();
	}
} )();