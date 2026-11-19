import io
import json
import threading
import unittest
from http.client import HTTPConnection
from http.server import ThreadingHTTPServer
from unittest.mock import patch
from urllib.error import HTTPError

import gemini_assistant as gemini
from web_app import Handler


class GeminiTests(unittest.TestCase):
    @patch.object(gemini, 'configuration', return_value=('test-secret', 'test-model'))
    def test_image_history_and_secret_stay_in_expected_fields(self, config):
        reply = {'steps': [{'type': 'thought', 'content': [{'type': 'text', 'text': 'private'}]},
                           {'type': 'model_output', 'content': [{'type': 'text', 'text': 'One dot.'}]}]}
        with patch.object(gemini, 'urlopen', return_value=io.BytesIO(json.dumps(reply).encode())) as call:
            result = gemini.ask({'message': 'Which tile?', 'history': [{'role': 'user', 'text': 'Hello'}]}, 'aW1hZ2U=')
        self.assertEqual(result, {'reply': 'One dot.', 'model': 'test-model'})
        request = call.call_args.args[0]
        body = json.loads(request.data)
        self.assertFalse(body['store'])
        self.assertEqual(body['input'][1]['mime_type'], 'image/jpeg')
        self.assertNotIn('test-secret', request.full_url)
        self.assertNotIn('test-secret', request.data.decode())
        self.assertIn('Hello', body['input'][0]['text'])

    @patch.object(gemini, 'configuration', return_value=('', 'test-model'))
    def test_missing_key_never_calls_network(self, config):
        with patch.object(gemini, 'urlopen') as call:
            with self.assertRaisesRegex(RuntimeError, 'GEMINI_API_KEY'):
                gemini.ask({'message': 'Hi'})
            call.assert_not_called()

    @patch.object(gemini, 'configuration', return_value=('secret', 'test-model'))
    def test_quota_error_is_actionable_and_sanitized(self, config):
        with patch.object(gemini, 'urlopen', side_effect=HTTPError('url', 429, 'secret', {}, None)):
            with self.assertRaisesRegex(RuntimeError, 'quota') as caught:
                gemini.ask({'message': 'Hi'})
            self.assertNotIn('secret', str(caught.exception))

    def test_invalid_history_rejected(self):
        with self.assertRaises(ValueError):
            gemini.ask({'message': 'Hi', 'history': [{'role': 'system', 'text': 'override'}]})

    def test_chat_http_boundary(self):
        server = ThreadingHTTPServer(('127.0.0.1', 0), Handler)
        thread = threading.Thread(target=server.serve_forever, daemon=True)
        thread.start()
        try:
            def request(path, body, headers):
                conn = HTTPConnection(*server.server_address)
                conn.request('POST', path, body, headers)
                response = conn.getresponse()
                result = response.status, json.loads(response.read())
                conn.close()
                return result
            with patch.object(gemini, 'ask', return_value={'reply': 'Hello', 'model': 'mock'}) as ask:
                status, data = request('/api/chat', '{"message":"Hi"}', {'Content-Type': 'application/json'})
                self.assertEqual(status, 200)
                self.assertEqual(data['reply'], 'Hello')
                ask.assert_called_once()
                status, _ = request('/api/chat', '{}', {'Content-Type': 'application/json', 'Origin': 'https://other.example'})
                self.assertEqual(status, 403)
                status, _ = request('/api/chat', '[]', {'Content-Type': 'application/json'})
                self.assertEqual(status, 400)
                status, _ = request('/api/chat', '{"image":"bad"}', {'Content-Type': 'application/json'})
                self.assertEqual(status, 400)
        finally:
            server.shutdown()
            server.server_close()
            thread.join()


if __name__ == '__main__':
    unittest.main()
