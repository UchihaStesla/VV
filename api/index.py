import os
import json
import logging
import subprocess
from flask import Flask, request, jsonify, Response
from duckduckgo_search import DDGS

logging.basicConfig(level=logging.DEBUG, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

app = Flask(__name__)

def generate_subtitle_search_stream(query_string):
    """Generates streaming JSON results from the Rust subtitle search API."""
    rust_process = None
    try:
        logger.debug(f"Starting Rust subprocess with query: {query_string}")
        rust_process = subprocess.Popen(
            ['./api/subtitle_search_api'], 
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            bufsize=1,
            encoding='utf-8'
        )

        rust_process.stdin.write(query_string + '\n')
        rust_process.stdin.flush()
        rust_process.stdin.close()

        first_item = True
        while True:
            line = rust_process.stdout.readline()
            if not line:
                logger.debug("Rust process stdout stream ended.")
                break 
                
            if not line.strip():
                continue

            if not first_item:
                yield '\n'
            first_item = False

            yield line.strip()

        stderr_output = rust_process.stderr.read()
        if stderr_output:
            logger.error(f"Rust process stderr output: {stderr_output.strip()}")
            
    except FileNotFoundError:
        logger.error("Error: Subtitle search executable './api/subtitle_search_api' not found.")
        yield json.dumps({
            "status": "error",
            "message": "Subtitle search executable not found."
        })
    except BrokenPipeError:
        logger.warning("Broken pipe error, possibly Rust process exited early.")
    except Exception as e:
        logger.error(f"Error during subtitle search stream: {e}", exc_info=True)
        yield json.dumps({
            "status": "error",
            "message": "搜索过程中发生内部错误"
        })
    finally:
        if rust_process:
            try:
                if not rust_process.stdin.closed:
                    rust_process.stdin.close()
                if rust_process.stdout:
                    rust_process.stdout.close()
                if rust_process.stderr:
                    rust_process.stderr.close()
                rust_process.terminate()
                rust_process.wait(timeout=5)
                logger.debug(f"Rust process terminated with return code: {rust_process.returncode}")
            except ProcessLookupError:
                 logger.debug("Rust process already finished.")
            except Exception as e_term:
                logger.error(f"Error terminating/cleaning up Rust process: {e_term}")


@app.route('/search', methods=['GET'])
def search():
    """
    Performs a search.
    If 'only_search=true', performs a DuckDuckGo web search ignoring other params.
    Otherwise, performs a subtitle search using the Rust API with filtering parameters.
    """
    query = request.args.get('query', '')
    only_search_str = request.args.get('only_search', 'false').lower()

    if not query:
        logger.warning("Search request received with empty query.")
        return jsonify({
            "status": "error",
            "message": "搜索关键词不能为空"
        }), 400

    if only_search_str == 'true':
        logger.info(f"Performing DuckDuckGo search for query: '{query}'")
        try:
            with DDGS() as ddgs:
                 search_results = list(ddgs.text(query))

            logger.info(f"DuckDuckGo search returned {len(search_results)} results for query: '{query}'")
            return jsonify(search_results)

        except Exception as e:
            logger.error(f"Error during DuckDuckGo search for query '{query}': {str(e)}", exc_info=True)
            return jsonify({
                "status": "error",
                "message": "An internal server error occurred while fetching DuckDuckGo search results.",
                "details": str(e)
            }), 500

    else:
        logger.info(f"Performing subtitle search for query: '{query}'")
        try:
            min_ratio_str = request.args.get('min_ratio', '50')
            min_similarity_str = request.args.get('min_similarity', '0.5')
            max_results_str = request.args.get('max_results', '50')

            try:
                min_ratio = float(min_ratio_str)
                min_similarity = float(min_similarity_str)
                max_results = int(max_results_str)

                if not (0 <= min_ratio <= 100):
                     raise ValueError("min_ratio must be between 0 and 100")
                if not (0 <= min_similarity <= 1):
                     raise ValueError("min_similarity must be between 0.0 and 1.0")
                if max_results <= 0:
                     raise ValueError("max_results must be a positive integer")

            except ValueError as ve:
                logger.warning(f"Invalid parameter format: {ve}")
                return jsonify({
                    "status": "error",
                    "message": f"参数格式错误: {ve}"
                }), 400

            query_string = f"query={query}"
            query_string += f"&min_ratio={min_ratio_str}"
            query_string += f"&min_similarity={min_similarity_str}"
            query_string += f"&max_results={max_results_str}"

            return Response(
                generate_subtitle_search_stream(query_string),
                mimetype='application/json',
                headers={
                    'X-Accel-Buffering': 'no',
                    'Cache-Control': 'no-cache'
                }
            )

        except Exception as e:
            logger.error(f"Unexpected error setting up subtitle search: {e}", exc_info=True)
            return jsonify({
                "status": "error",
                "message": "服务器内部错误"
            }), 500

@app.route('/health', methods=['GET'])
def health_check():
    return jsonify({"status": "ok"})

@app.route('/')
def index():
    return jsonify({
        "message": "Search API is running. Use /search endpoint.",
        "endpoints": {
            "/search": {
                "description": "Performs subtitle search or DuckDuckGo search.",
                "params": {
                    "query": "(string, required) Search term.",
                    "only_search": "(boolean, optional, default=false) If true, performs DuckDuckGo search and ignores other params.",
                    "min_ratio": "(float, optional, default=50) Minimum ratio for subtitle search (0-100).",
                    "min_similarity": "(float, optional, default=0.5) Minimum similarity for subtitle search (0.0-1.0).",
                    "max_results": "(int, optional, default=50) Max results for subtitle search."
                }
            },
            "/health": "Returns API health status."
        }
    })

application = app

if __name__ == "__main__":
    app.run(debug=True, host='0.0.0.0', port=8000)