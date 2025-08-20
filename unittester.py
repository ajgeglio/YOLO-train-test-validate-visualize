import unittest
from unittest.mock import patch, mock_open, MagicMock
from src import reports, predicting
import os, glob
import datetime
import torch, argparse  # Import necessary modules from your script
from ultralytics import YOLO
from PIL import Image
from timeit import default_timer as stopwatch

class TestYOLOInference(unittest.TestCase):
    
    def setUp(self):
        self.start_time = stopwatch()
        torch.cuda.empty_cache()
        self.now = datetime.datetime.now()
        self.test_output = "unittester_output"
        self.test_dir = "test_images"
        self.test_csv = "test_image_list.csv"
        self.weights_path = "yolov8x.pt"
    
    @patch('argparse.ArgumentParser.parse_args')
    def test_argparse(self, mock_parse_args):
        mock_parse_args.return_value = argparse.Namespace(
            has_labels=True,
            img_directory=self.test_dir,
            img_list_csv=self.test_csv,
            lbl_list_csv=None,
            lbl_database=None,
            weights=self.weights_path,
            start_batch=0,
            plot=False,
            supress_log=True,
            output_name=self.test_output,
            batch_size=4,
            img_size=2048,
            iou=0.6,
            confidence=0.01,
            verify=False
        )

        args = argparse.ArgumentParser(description="Test").parse_args()

        self.assertEqual(args.has_labels, True)
        self.assertEqual(args.img_directory, self.test_dir)
        self.assertEqual(args.weights, self.weights_path)
        self.assertEqual(args.batch_size, 4)
        self.assertEqual(args.img_size, 2048)
        self.assertEqual(args.iou, 0.6)
        self.assertEqual(args.confidence, 0.01)

    @patch('glob.glob')
    @patch('builtins.open', new_callable=mock_open, read_data='test_image1.jpg\ntest_image2.jpg\n')
    def test_image_loading(self, mock_file, mock_glob):
        mock_glob.side_effect = [['test_image1.jpg'], ['test_image2.png'], ['test_image3.tif']]

        test_images = glob.glob(f'{self.test_dir}/*.[jJ][pP][gG]') + \
                      glob.glob(f'{self.test_dir}/*.[pP][nN][gG]') + \
                      glob.glob(f'{self.test_dir}/*.[tT][iI][fF]')
        expected_images = ['test_image1.jpg', 'test_image2.png', 'test_image3.tif']
        self.assertEqual(sorted(test_images), sorted(expected_images))
        
        with open(self.test_csv, 'r') as f:
            img_list = f.read().splitlines()
        expected_list = ['test_image1.jpg', 'test_image2.jpg']
        self.assertEqual(img_list, expected_list)

    @patch('src.predicting.YOLO_predict_w_outut')
    @patch('ultralytics.YOLO')
    def test_yolo_model(self, mock_yolo, mock_predict):
        model = mock_yolo.return_value
        model.return_value = [MagicMock()] * 4  # Mocking 4 results for the batch
        imgs = ["test_image1.jpg", "test_image2.jpg", "test_image3.jpg", "test_image4.jpg"]
        lbls = ["test_label1.txt", "test_label2.txt", "test_label3.txt", "test_label4.txt"]
        
        results = model(imgs, stream=True, half=True, iou=0.6, conf=0.01, imgsz=2048, classes=[0])
        
        for r, lbl, img_path in zip(results, lbls, imgs):
            predicting.YOLO_predict_w_outut(r, lbl, img_path, "predictions.csv", "labels.csv", None, False, True)
        
        self.assertTrue(mock_predict.called)

if __name__ == '__main__':
    unittest.main()