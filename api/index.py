from http.server import BaseHTTPRequestHandler
import os

class handler(BaseHTTPRequestHandler):
    def do_GET(self):
        if 'predictions' in self.path:
            if os.path.exists('predictions.csv'):
                self.send_response(200)
                self.send_header('Content-type', 'text/csv')
                self.send_header('Content-Disposition', 'attachment; filename="predictions.csv"')
                self.end_headers()
                with open('predictions.csv', 'rb') as f:
                    self.wfile.write(f.read())
            else:
                self.send_response(404)
                self.send_header('Content-type', 'text/plain')
                self.end_headers()
                self.wfile.write('predictions.csv not found'.encode('utf-8'))
        else:
            self.send_response(200)
            self.send_header('Content-type', 'text/html')
            self.end_headers()
            html = """
            <!DOCTYPE html>
            <html>
            <head><title>Nexora MLOps</title></head>
            <body style="font-family: sans-serif; text-align: center; margin-top: 50px;">
                <h1>Nexora MLOps Pipeline</h1>
                <p>Status: <strong style="color: green;">Live</strong></p>
                <p>The MLOps pipeline is active. Predictions are generated via GitHub Actions.</p>
                <p><a href="/api/index?predictions">Download latest predictions.csv</a></p>
            </body>
            </html>
            """
            self.wfile.write(html.encode('utf-8'))
        return
