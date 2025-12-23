import pytest
import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from config import app, db
from db_src.DB_MODEL import User

@pytest.fixture
def client():
    app.config['TESTING'] = True
    app.config['SQLALCHEMY_DATABASE_URI'] = 'sqlite:///:memory:'
    app.config['WTF_CSRF_ENABLED'] = False

    with app.test_client() as client:
        with app.app_context():
            # Mock heavy ML modules to prevent loading them during tests
            from unittest.mock import MagicMock
            sys.modules['ai_grader.perform_classification'] = MagicMock()
            sys.modules['ai_grader.generate_caption'] = MagicMock()
            sys.modules['ai_grader.detect_opacity'] = MagicMock()
            sys.modules['ai_grader.detect_external_devices'] = MagicMock()
            sys.modules['ai_grader.generate_segmentation'] = MagicMock()
            sys.modules['ai_grader.generate_heatmaps'] = MagicMock()
            
            import main
            db.create_all()
            yield client
            db.session.remove()
            db.drop_all()

@pytest.fixture
def init_database():
    with app.app_context():
        db.create_all()
        yield db
        db.session.remove()
        db.drop_all()