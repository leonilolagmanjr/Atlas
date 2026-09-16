import unittest

from fastapi.testclient import TestClient

from api import app


class ApiSmokeTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.client = TestClient(app)

    def test_health_reports_local_runtime(self):
        response = self.client.get("/api/health")

        self.assertEqual(response.status_code, 200)
        self.assertTrue(response.json()["local"])
        self.assertEqual(response.json()["status"], "online")

    def test_system_reports_real_machine_data(self):
        response = self.client.get("/api/system")

        self.assertEqual(response.status_code, 200)
        self.assertIn("system", response.json())
        self.assertIn("disk", response.json()["system"])


if __name__ == "__main__":
    unittest.main()
