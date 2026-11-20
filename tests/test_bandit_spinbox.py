# coding:utf-8
"""BandIt Plus overlap and segment controls can be edited and saved."""

import os
import sys
import types
import unittest

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

if "faster_whisper_GUI" not in sys.modules:
    package = types.ModuleType("faster_whisper_GUI")
    package.__path__ = [os.path.join(ROOT, "faster_whisper_GUI")]
    package.__package__ = "faster_whisper_GUI"
    sys.modules["faster_whisper_GUI"] = package

from PySide6.QtCore import QPoint, Qt
from PySide6.QtTest import QTest
from PySide6.QtWidgets import QApplication

from faster_whisper_GUI.banditPageNavigationInterface import BanditPageNavigation


class BanditSpinBoxTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.app = QApplication.instance() or QApplication([])

    def setUp(self):
        self.page = BanditPageNavigation()
        self.page.resize(1000, 900)
        self.page.show()
        self.app.processEvents()
        self.page.setParam({"overlap": 0.1, "segment": 7.8, "tracks": 4})
        self.app.processEvents()
        self.panel = self.page.bandit_param_widget

    def tearDown(self):
        self.page.deleteLater()
        self.app.processEvents()

    def test_saved_values_are_visible_and_editable(self):
        overlap = self.panel.spinBox_overlap
        segment = self.panel.spinBox_segment
        stems = self.panel.comboBox_stems

        self.assertTrue(overlap.isEnabled())
        self.assertFalse(overlap.isReadOnly())
        self.assertGreater(overlap.lineEdit().width(), 40)
        self.assertEqual(overlap.lineEdit().text(), "0.10")
        self.assertEqual(overlap.value(), 0.1)
        self.assertLessEqual(overlap.minimum(), 0.0)
        self.assertGreaterEqual(overlap.maximum(), 0.5)

        self.assertTrue(segment.isEnabled())
        self.assertFalse(segment.isReadOnly())
        self.assertGreater(segment.lineEdit().width(), 40)
        self.assertEqual(segment.lineEdit().text(), "7.8")
        self.assertEqual(segment.value(), 7.8)
        self.assertLessEqual(segment.minimum(), 6.0)
        self.assertGreaterEqual(segment.maximum(), 10.0)

        self.assertEqual(stems.currentIndex(), 4)
        self.assertTrue(stems.isEnabled())

        QTest.mouseClick(overlap.lineEdit(), Qt.MouseButton.LeftButton, pos=QPoint(30, 12))
        self.assertTrue(overlap.lineEdit().hasFocus())
        overlap.lineEdit().selectAll()
        QTest.keyClicks(overlap.lineEdit(), "0.35")
        overlap.interpretText()

        QTest.mouseClick(segment.upButton, Qt.MouseButton.LeftButton)
        self.assertEqual(self.page.getParam(), {"overlap": 0.35, "segment": 8.8, "tracks": 4})

        self.page.setParam({"overlap": 0.1, "segment": 7.8, "tracks": 1})
        self.assertEqual(self.page.getParam(), {"overlap": 0.1, "segment": 7.8, "tracks": 1})


if __name__ == "__main__":
    unittest.main()
