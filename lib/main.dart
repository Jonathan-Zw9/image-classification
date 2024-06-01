// ignore_for_file: unnecessary_null_comparison

import 'dart:async';
import 'dart:io';
import 'dart:math';
// ignore: unnecessary_import
import 'dart:typed_data';
import 'package:flutter/foundation.dart';
import 'package:flutter/material.dart';
import 'package:flutter/services.dart';
import 'package:image/image.dart' as img;

import 'package:tflite_v2/tflite_v2.dart';
import 'package:image_picker/image_picker.dart';

void main() => runApp(const App());

const String mobile = "MobileNet";
const String ssd = "SSD MobileNet";
const String yolo = "Tiny YOLOv2";
const String deeplab = "DeepLab";
const String posenet = "PoseNet";

class App extends StatelessWidget {
  const App({super.key});

  @override
  Widget build(BuildContext context) {
    return const MaterialApp(
      home: MyApp(),
    );
  }
}

class MyApp extends StatefulWidget {
  const MyApp({super.key});

  @override
  // ignore: library_private_types_in_public_api
  _MyAppState createState() => _MyAppState();
}

class _MyAppState extends State<MyApp> {
  late File _image;
  late List _recognitions;
  String _model = mobile;
  late double _imageHeight;
  late double _imageWidth;
  bool _busy = false;

  Future predictImagePicker() async {
    var imagePicker = ImagePicker();
    var image = await imagePicker.pickImage(source: ImageSource.gallery);
    if (image == null) {
      setState(() {
        _busy = false;
        // Handle no image selected case, possibly setting _image to a default
      });
      return;
    }
    setState(() {
      _busy = true;
    });
    predictImage(File(image.path));
  }

  Future predictImage(File image) async {
    if (image == null) return;

    switch (_model) {
      case yolo:
        await yolov2Tiny(image);
        break;
      case ssd:
        await ssdMobileNet(image);
        break;
      case deeplab:
        await segmentMobileNet(image);
        break;
      case posenet:
        await poseNet(image);
        break;
      default:
        await recognizeImage(image);
      // await recognizeImageBinary(image);
    }

    FileImage(image)
        .resolve(const ImageConfiguration())
        .addListener(ImageStreamListener((ImageInfo info, bool _) {
      setState(() {
        _imageHeight = info.image.height.toDouble();
        _imageWidth = info.image.width.toDouble();
      });
    }));

    setState(() {
      _image = image;
      _busy = false;
    });
  }

  @override
  void initState() {
    super.initState();
    _busy = true;
    loadModel().then((_) {
      if (_image != null) {
        predictImage(_image);
      } else {
        setState(() {
          _busy = false;
        });
      }
    });
  }

  Future loadModel() async {
    Tflite.close();
    try {
      String? res;
      switch (_model) {
        case yolo:
          res = await Tflite.loadModel(
            model: "assets/model_unquant.tflite",
            labels: "assets/labels.txt",
            // useGpuDelegate: true,
          );
          break;
        case ssd:
          res = await Tflite.loadModel(
            model: "assets/model_unquant.tflite",
            labels: "assets/labels.txt",
            // useGpuDelegate: true,
          );
          break;
        case deeplab:
          res = await Tflite.loadModel(
            model: "assets/model_unquant.tflite",
            labels: "assets/labels.txt",
            // useGpuDelegate: true,
          );
          break;
        case posenet:
          res = await Tflite.loadModel(
            model: "assets/model_unquant.tflite",
            // useGpuDelegate: true,
          );
          break;
        default:
          res = await Tflite.loadModel(
            model: "assets/model_unquant.tflite",
            labels: "assets/labels.txt",
            // useGpuDelegate: true,
          );
      }
      if (kDebugMode) {
        print(res);
      }
    } on PlatformException {
      if (kDebugMode) {
        print('Failed to load model.');
      }
    }
  }

  Uint8List imageToByteListFloat32(
      img.Image image, int inputSize, double mean, double std) {
    final convertedBytes = Float32List(1 * inputSize * inputSize * 3);
    final buffer = Float32List.view(convertedBytes.buffer);
    int pixelIndex = 0;
    for (var i = 0; i < inputSize; i++) {
      for (var j = 0; j < inputSize; j++) {
        final pixel = image.getPixel(j, i);
        buffer[pixelIndex++] = (pixel.r - mean) / std;
        buffer[pixelIndex++] = (pixel.g - mean) / std;
        buffer[pixelIndex++] = (pixel.b - mean) / std;
      }
    }
    return convertedBytes.buffer.asUint8List();
  }

  Uint8List imageToByteListUint8(img.Image image, int inputSize) {
    final convertedBytes = Uint8List(1 * inputSize * inputSize * 3);
    final buffer = Uint8List.view(convertedBytes.buffer);
    int pixelIndex = 0;
    for (var i = 0; i < inputSize; i++) {
      for (var j = 0; j < inputSize; j++) {
        final pixel = image.getPixel(j, i);
        buffer[pixelIndex++] = pixel.r.toInt();
        buffer[pixelIndex++] = pixel.g.toInt();
        buffer[pixelIndex++] = pixel.b.toInt();
      }
    }
    return convertedBytes.buffer.asUint8List();
  }

  Future recognizeImage(File image) async {
    int startTime = DateTime.now().millisecondsSinceEpoch;
    var recognitions = await Tflite.runModelOnImage(
      path: image.path,
      numResults: 6,
      threshold: 0.05,
      imageMean: 127.5,
      imageStd: 127.5,
    );
    setState(() {
      _recognitions = recognitions!;
    });
    int endTime = DateTime.now().millisecondsSinceEpoch;
    if (kDebugMode) {
      print("Inference took ${endTime - startTime}ms");
    }
  }

  Future recognizeImageBinary(File image) async {
    int startTime = DateTime.now().millisecondsSinceEpoch;
    var imageBytes = (await rootBundle.load(image.path)).buffer;
    img.Image? oriImage = img.decodeJpg(imageBytes.asUint8List());
    img.Image resizedImage = img.copyResize(oriImage!, height: 224, width: 224);
    var recognitions = await Tflite.runModelOnBinary(
      binary: imageToByteListFloat32(resizedImage, 224, 127.5, 127.5),
      numResults: 6,
      threshold: 0.05,
    );
    setState(() {
      _recognitions = recognitions!;
    });
    int endTime = DateTime.now().millisecondsSinceEpoch;
    if (kDebugMode) {
      print("Inference took ${endTime - startTime}ms");
    }
  }

  Future yolov2Tiny(File image) async {
    int startTime = DateTime.now().millisecondsSinceEpoch;
    var recognitions = await Tflite.detectObjectOnImage(
      path: image.path,
      model: "YOLO",
      threshold: 0.3,
      imageMean: 0.0,
      imageStd: 255.0,
      numResultsPerClass: 1,
    );
    
    setState(() {
      _recognitions = recognitions!;
    });
    int endTime = DateTime.now().millisecondsSinceEpoch;
    if (kDebugMode) {
      print("Inference took ${endTime - startTime}ms");
    }
  }

  Future ssdMobileNet(File image) async {
    int startTime = DateTime.now().millisecondsSinceEpoch;
    var recognitions = await Tflite.detectObjectOnImage(
      path: image.path,
      numResultsPerClass: 1,
    );
   
    setState(() {
      _recognitions = recognitions!;
    });
    int endTime = DateTime.now().millisecondsSinceEpoch;
    if (kDebugMode) {
      print("Inference took ${endTime - startTime}ms");
    }
  }

  Future segmentMobileNet(File image) async {
    int startTime = DateTime.now().millisecondsSinceEpoch;
    var recognitions = await Tflite.runSegmentationOnImage(
      path: image.path,
      imageMean: 127.5,
      imageStd: 127.5,
    );

    setState(() {
      _recognitions = recognitions!;
    });
    int endTime = DateTime.now().millisecondsSinceEpoch;
    if (kDebugMode) {
      print("Inference took ${endTime - startTime}");
    }
  }

  Future poseNet(File image) async {
    int startTime = DateTime.now().millisecondsSinceEpoch;
    var recognitions = await Tflite.runPoseNetOnImage(
      path: image.path,
      numResults: 2,
    );

    if (kDebugMode) {
      print(recognitions);
    }

    setState(() {
      _recognitions = recognitions!;
    });
    int endTime = DateTime.now().millisecondsSinceEpoch;
    if (kDebugMode) {
      print("Inference took ${endTime - startTime}ms");
    }
  }

  onSelect(model) async {
    setState(() {
      _busy = true;
      _model = model;
      _recognitions = [];
    });
    await loadModel();

    if (_image != null) {
      predictImage(_image);
    } else {
      setState(() {
        _busy = false;
      });
    }
  }

  List<Widget> renderBoxes(Size screen) {
    if (_recognitions == null) return [];
    if (_imageHeight == null || _imageWidth == null) return [];

    double factorX = screen.width;
    double factorY = _imageHeight / _imageWidth * screen.width;
    Color blue = const Color.fromRGBO(37, 213, 253, 1.0);
    return _recognitions.map((re) {
      return Positioned(
        left: re["rect"]["x"] * factorX,
        top: re["rect"]["y"] * factorY,
        width: re["rect"]["w"] * factorX,
        height: re["rect"]["h"] * factorY,
        child: Container(
          decoration: BoxDecoration(
            borderRadius: const BorderRadius.all(Radius.circular(8.0)),
            border: Border.all(
              color: blue,
              width: 2,
            ),
          ),
          child: Text(
            "${re["detectedClass"]} ${(re["confidenceInClass"] * 100).toStringAsFixed(0)}%",
            style: TextStyle(
              background: Paint()..color = blue,
              color: Colors.white,
              fontSize: 12.0,
            ),
          ),
        ),
      );
    }).toList();
  }

  List<Widget> renderKeypoints(Size screen) {
    if (_recognitions == null) return [];
    if (_imageHeight == null || _imageWidth == null) return [];

    double factorX = screen.width;
    double factorY = _imageHeight / _imageWidth * screen.width;

    var lists = <Widget>[];
    for (var re in _recognitions) {
      var color = Color((Random().nextDouble() * 0xFFFFFF).toInt() << 0)
          .withOpacity(1.0);
      var list = re["keypoints"].values.map<Widget>((k) {
        return Positioned(
          left: k["x"] * factorX - 6,
          top: k["y"] * factorY - 6,
          width: 100,
          height: 12,
          child: Text(
            "● ${k["part"]}",
            style: TextStyle(
              color: color,
              fontSize: 12.0,
            ),
          ),
        );
      }).toList();

      lists.addAll(list);
    }

    return lists;
  }

  @override
  Widget build(BuildContext context) {
    Size size = MediaQuery.of(context).size;
    List<Widget> stackChildren = [];

    if (_model == deeplab && _recognitions != null) {
      stackChildren.add(
        Positioned(
          top: 0.0,
          left: 0.0,
          width: size.width,
          child: _image == null
              ? const Text('No image selected.')
              : Container(
                  decoration: BoxDecoration(
                    image: DecorationImage(
                      alignment: Alignment.topCenter,
                      image: MemoryImage(
                          Uint8List.fromList(_recognitions.cast<int>())),
                      fit: BoxFit.fill,
                    ),
                  ),
                  child: Opacity(
                    opacity: 0.3,
                    child: Image.file(_image),
                  ),
                ),
        ),
      );
    } else {
      stackChildren.add(
        Positioned(
          top: 0.0,
          left: 0.0,
          width: size.width,
          child: _image == null
              ? const Text('No image selected.')
              : Image.file(_image),
        ),
      );
    }

    if (_model == mobile) {
      stackChildren.add(Center(
        child: Column(
          children: _recognitions != null
              ? _recognitions.map((res) {
                  return Text(
                    "${res["index"]} - ${res["label"]}: ${res["confidence"].toStringAsFixed(3)}",
                    style: TextStyle(
                      color: Colors.black,
                      fontSize: 20.0,
                      background: Paint()..color = Colors.white,
                    ),
                  );
                }).toList()
              : [],
        ),
      ));
    } else if (_model == ssd || _model == yolo) {
      stackChildren.addAll(renderBoxes(size));
    } else if (_model == posenet) {
      stackChildren.addAll(renderKeypoints(size));
    }

    if (_busy) {
      stackChildren.add(const Opacity(
        opacity: 0.3,
        child: ModalBarrier(dismissible: false, color: Colors.grey),
      ));
      stackChildren.add(const Center(child: CircularProgressIndicator()));
    }

    return Scaffold(
      appBar: AppBar(
        title: const Text('tflite example app'),
        actions: <Widget>[
          PopupMenuButton<String>(
            onSelected: onSelect,
            itemBuilder: (context) {
              List<PopupMenuEntry<String>> menuEntries = [
                const PopupMenuItem<String>(
                  value: mobile,
                  child: Text(mobile),
                ),
                const PopupMenuItem<String>(
                  value: ssd,
                  child: Text(ssd),
                ),
                const PopupMenuItem<String>(
                  value: yolo,
                  child: Text(yolo),
                ),
                const PopupMenuItem<String>(
                  value: deeplab,
                  child: Text(deeplab),
                ),
                const PopupMenuItem<String>(
                  value: posenet,
                  child: Text(posenet),
                )
              ];
              return menuEntries;
            },
          )
        ],
      ),
      body: Stack(
        children: stackChildren,
      ),
      floatingActionButton: FloatingActionButton(
        onPressed: predictImagePicker,
        tooltip: 'Pick Image',
        child: const Icon(Icons.image),
      ),
    );
  }
}
