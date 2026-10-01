"""
test_app.py
Verification suite using streamlit.testing.v1.AppTest
Verifies:
1. Zero exceptions on initial app load.
2. All 4 navigation tabs render cleanly.
3. Prediction runs with zero exceptions upon clicking Predict button and persists in state.
4. Threshold reset callback executes cleanly.
5. Batch CSV upload scores sample customers with zero exceptions.
"""

import os
import io
import pandas as pd
from streamlit.testing.v1 import AppTest

APP_PATH = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "app.py"))


def test_app_load_and_tabs():
    """Verify zero exceptions on initial load and that tabs render."""
    at = AppTest.from_file(APP_PATH, default_timeout=30)
    at.run()
    
    # Assert zero exceptions
    assert len(at.exception) == 0, f"App threw exceptions on load: {[e.value for e in at.exception]}"
    print("PASS: App loaded with 0 exceptions.")
    
    # Check that tabs rendered
    tab_labels = [tab.label for tab in at.tabs]
    assert len(tab_labels) == 4, f"Expected 4 tabs, found: {tab_labels}"
    print(f"PASS: All 4 tabs rendered: {tab_labels}")


def test_single_prediction_flow():
    """Verify prediction button triggers cleanly and state persists."""
    at = AppTest.from_file(APP_PATH, default_timeout=30)
    at.run()
    assert len(at.exception) == 0
    
    # Find and click the predict button
    predict_btn = next((b for b in at.button if "Predict Campaign Response" in b.label), None)
    assert predict_btn is not None, "Predict button not found in app"
    
    predict_btn.click().run()
    assert len(at.exception) == 0, f"Exception after clicking predict: {[e.value for e in at.exception]}"
    
    # Verify session state contains prediction
    assert at.session_state.single_prediction is not None, "Prediction was not stored in session_state"
    assert "proba" in at.session_state.single_prediction
    assert "pred_class" in at.session_state.single_prediction
    
    proba = at.session_state.single_prediction['proba']
    pred = at.session_state.single_prediction['pred_class']
    print(f"PASS: Single customer prediction executed cleanly (Proba: {proba:.4f}, Class: {pred}).")


def test_threshold_slider_and_reset():
    """Verify threshold slider and reset callback."""
    at = AppTest.from_file(APP_PATH, default_timeout=30)
    at.run()
    assert len(at.exception) == 0
    
    # Adjust sidebar slider value
    slider = at.sidebar.slider[0]
    initial_val = slider.value
    slider.set_value(0.42).run()
    assert len(at.exception) == 0
    assert abs(at.session_state.threshold_slider - 0.42) < 1e-4
    
    # Click reset button in sidebar
    reset_btn = next((b for b in at.sidebar.button if "Reset" in b.label), None)
    assert reset_btn is not None, "Reset button not found in sidebar"
    reset_btn.click().run()
    assert len(at.exception) == 0
    assert abs(at.session_state.threshold_slider - initial_val) < 1e-4, f"Threshold did not reset: {at.session_state.threshold_slider}"
    print(f"PASS: Reset threshold callback succeeded (Reset to {initial_val:.2f}).")


def test_batch_csv_upload():
    """Verify batch scoring with sample CSV upload."""
    at = AppTest.from_file(APP_PATH, default_timeout=30)
    at.run()
    assert len(at.exception) == 0
    
    # Create sample CSV in memory with missing values and edge cases to test audit warnings
    sample_df = pd.DataFrame([
        {
            'age_group': '26-35',
            'income': 58000.0,
            'previous_purchases': 10,
            'purchase_frequency': 3.2,
            'previous_campaign_response': 1,
            'website_visits': 12,
            'email_engagement': 0.75,
            'discount_usage': 0.60
        },
        {
            'age_group': None, # Missing categorical
            'income': None, # Missing numeric
            'previous_purchases': 2,
            'purchase_frequency': 0.5,
            'previous_campaign_response': 0,
            'website_visits': 3,
            'email_engagement': 0.15,
            'discount_usage': 0.20
        },
        {
            'age_group': 'UnknownGroup', # Out-of-bounds categorical
            'income': 95000.0,
            'previous_purchases': 15,
            'purchase_frequency': 1.8,
            'previous_campaign_response': 2, # Out-of-bounds binary
            'website_visits': 6,
            'email_engagement': 1.25, # Out-of-bounds proportion
            'discount_usage': 0.45
        }
    ])
    csv_bytes = sample_df.to_csv(index=False).encode('utf-8')
    
    uploader = at.file_uploader[0]
    uploader.upload(filename="batch_test.csv", content=csv_bytes, mime_type="text/csv").run()
    
    assert len(at.exception) == 0, f"Exception on batch CSV upload: {[e.value for e in at.exception]}"
    
    # Verify success or warning message rendered
    success_or_warning = len(at.success) > 0 or len(at.warning) > 0
    assert success_or_warning, "Expected batch completion message or warning"
    print("PASS: Batch CSV upload executed cleanly with audit checks and zero exceptions.")


if __name__ == '__main__':
    print("=" * 60)
    print("RUNNING STREAMLIT AppTest VERIFICATION SUITE")
    print("=" * 60)
    test_app_load_and_tabs()
    test_single_prediction_flow()
    test_threshold_slider_and_reset()
    test_batch_csv_upload()
    print("=" * 60)
    print("ALL APP TESTS PASSED WITH 0 EXCEPTIONS!")
    print("=" * 60)
