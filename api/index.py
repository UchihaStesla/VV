import os
import json
import logging
import subprocess
from flask import Flask, request, jsonify, send_file
from typing import List, Tuple

logging.basicConfig(level=logging.DEBUG)



app = Flask(__name__)
@app.route('/search', methods=['GET'])
def search():
    try:
        query = request.args.get('query', '')
        min_ratio = request.args.get('min_ratio', '50')
        min_similarity = request.args.get('min_similarity', '0.5')
        max_results = request.args.get('max_results', '50')

        # 添加参数验证
        if not query:
            return jsonify({
                "status": "error",
                "message": "搜索关键词不能为空"
            }), 400

        try:
            min_ratio = float(min_ratio)
            min_similarity = float(min_similarity)
            if not (0 <= min_ratio <= 100) or not (0 <= min_similarity <= 1):
                raise ValueError()
        except ValueError:
            return jsonify({
                "status": "error",
                "message": "参数格式错误"
            }), 400

        query_string = f"query={query}"
        query_string += f"&min_ratio={min_ratio}"
        query_string += f"&min_similarity={min_similarity}"
        query_string += f"&max_results={max_results}"

        def generate():
            try:
                rust_process = subprocess.Popen(
                    ['./api/subtitle_search_api'],
                    stdin=subprocess.PIPE,
                    stdout=subprocess.PIPE,
                    stderr=subprocess.PIPE,
                    text=True,
                    bufsize=1
                )
                
                rust_process.stdin.write(query_string + '\n')
                rust_process.stdin.flush()
                
                first_item = True
                while True:
                    line = rust_process.stdout.readline()
                    if not line:
                        break
                        
                    if not first_item:
                        yield '\n'
                    first_item = False
                    
                    yield line.strip()
            except Exception as e:
                logging.error(f"Stream error: {e}")
                yield json.dumps({
                    "status": "error",
                    "message": "搜索过程中发生错误"
                })
            finally:
                try:
                    rust_process.terminate()
                except:
                    pass

        return app.response_class(
            generate(),
            mimetype='application/json',
            headers={
                'X-Accel-Buffering': 'no',
                'Cache-Control': 'no-cache'
            }
        )

    except Exception as e:
        logging.error(f"Unexpected error: {e}")
        return jsonify({
            "status": "error",
            "message": "服务器内部错误"
        }), 500


application = app

if __name__ == "__main__":
    app.run(debug=True, host='0.0.0.0', port=8000)
