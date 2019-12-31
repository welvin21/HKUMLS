# Text Detection

## Usage

After installing all the required modules by `pip install -r requirements.txt`, make sure you are in the root directory `./FaceDetection`.

### Image
Run the script via command line by writing : <br/><br/>
`python image.py -m <path to caffe model> -p <path to .prototxt file> -i <path to an image file>` <br/><br/>

### Video
Run the script via command line by writing : <br/><br/>
`python video.py -m <path to caffe model> -p <path to .prototxt file>` <br/><br/>

`model.caffemodel` is a pre-trained model available in the root directory. <br/><br/>
`deploy.prototxt.txt` is a .prototxt file available in the root directory.

## Examples

![](https://github.com/welvin21/HKUMLS/blob/master/FaceDetection/assets/sample3.png?raw=true)
![](https://github.com/welvin21/HKUMLS/blob/master/FaceDetection/assets/sample3_prediction.png?raw=true)<br/><br/>
![](https://github.com/welvin21/HKUMLS/blob/master/FaceDetection/assets/sample4.png?raw=true)
![](https://github.com/welvin21/HKUMLS/blob/master/FaceDetection/assets/sample4_prediction.png?raw=true)<br/><br/>
