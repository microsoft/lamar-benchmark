//
//  IO.swift
//  ScanCapture
//
//  Created by Paul-Edouard Sarlin on 30.05.21.
//

import Foundation
import AVFoundation
import os.log
import ARKit
import MobileCoreServices  // for ImageI/O
import CoreMotion
import CoreBluetooth


func timestampToInt(_ timestamp: TimeInterval) -> Int64 {
    return Int64(round(timestamp * 1000000))  // second to microsecond
}


class ImageStreamer {
    var path: URL!
    var isInitialized: Bool = false
    let timeScale: Int32 = Int32(timestampToInt(1.0))  // microseconds
    private var _assetWriter: AVAssetWriter?
    private var _assetWriterInput: AVAssetWriterInput?
    private var _adapter: AVAssetWriterInputPixelBufferAdaptor?
    var counter: Int64 = 0
    
    init?(outDir: URL) {
        path = outDir.appendingPathComponent("images.mp4")
    }

    func initializeStream(buffer: CVPixelBuffer, timestamp: TimeInterval) {
        let writer = try! AVAssetWriter(outputURL: path, fileType: .mp4)
        writer.movieFragmentInterval = CMTimeMake(value: timestampToInt(1.0), timescale: timeScale)
        let settings: [String: Any] = [
            AVVideoCodecKey: AVVideoCodecType.h264,
            AVVideoWidthKey: CVPixelBufferGetWidthOfPlane(buffer, 0),
            AVVideoHeightKey: CVPixelBufferGetHeightOfPlane(buffer, 0),
            AVVideoCompressionPropertiesKey: [
                AVVideoProfileLevelKey: AVVideoProfileLevelH264HighAutoLevel,
            ],
        ]
        let input = AVAssetWriterInput(mediaType: .video, outputSettings: settings)
        input.mediaTimeScale = timeScale
        input.expectsMediaDataInRealTime = true
        let adapter = AVAssetWriterInputPixelBufferAdaptor(assetWriterInput: input, sourcePixelBufferAttributes: nil)
        if writer.canAdd(input) {
            writer.add(input)
        } else {
            os_log("Cannot initialize image stream", type:.error)
        }
        writer.startWriting()
        writer.startSession(atSourceTime: CMTimeMake(value: timestampToInt(timestamp), timescale:timeScale))
        
        _assetWriter = writer
        _assetWriterInput = input
        _adapter = adapter
        isInitialized = true
        counter = 0
    }
    
    func resetStream() {
        isInitialized = false
        _assetWriter = nil
        _assetWriterInput = nil
        _adapter = nil
    }
    
    func write(buffer: CVPixelBuffer, timestamp: TimeInterval) {
        if !isInitialized {
            initializeStream(buffer: buffer, timestamp: timestamp)
        }
        while _assetWriterInput?.isReadyForMoreMediaData == false {
            RunLoop.current.run(until: Date(timeIntervalSinceNow: 0.005))  // wait 5ms
        }
        let time = CMTimeMake(value: timestampToInt(timestamp), timescale: timeScale)
        if !_adapter!.append(buffer, withPresentationTime: time) {
            os_log("Could not append new image %@ to stream: %@ ", type:.error, counter, _assetWriter!.status.rawValue)
        }
        counter += 1
    }
    
    func finish() {
        os_log("Finishing the image stream, status: %d, ready? %d", _assetWriter!.status.rawValue, _assetWriterInput!.isReadyForMoreMediaData ? 1 : 0)
        _assetWriterInput?.markAsFinished()
        _assetWriter?.finishWriting { [weak self] in
            self?.resetStream()
        }
    }
}


class ImageWriter {
    var outDir: URL!
    let imageContext = CIContext(mtlDevice: MTLCreateSystemDefaultDevice()!)
    //    let imageContext = CIContext(options: nil)
    
    init?(outDir: URL) {
        self.outDir = outDir
        do {
            try FileManager.default.createDirectory(at: outDir, withIntermediateDirectories: true, attributes: nil)
        } catch {
            os_log("Cannot create the image directory: %@", type:.error, error.localizedDescription)
            return nil
        }
    }
    
    private func imageBufferToUIImage(buffer: CVPixelBuffer) -> UIImage {
        let ciImage = CIImage(cvPixelBuffer: buffer)
        let cgImage = self.imageContext.createCGImage(ciImage, from: ciImage.extent)
        let image = UIImage(cgImage: cgImage!)
        return image
    }

    func write(buffer: CVPixelBuffer, timestamp: TimeInterval) {
        let image = self.imageBufferToUIImage(buffer: buffer)
        if let data = image.jpegData(compressionQuality: 0.5) {
            let imagePath = outDir.appendingPathComponent(String(format: "%lld.jpg", timestampToInt(timestamp)))
            do {
                try data.write(to: imagePath)
            } catch {
                os_log("Cannot write image: %@", type:.error, error.localizedDescription)
            }
        }
    }
    
    // Apparently slower than using UIImage.jpegData
    func write2(buffer: CVPixelBuffer, timestamp: TimeInterval) {
        let ciImage = CIImage(cvPixelBuffer: buffer)
        let cgImage = self.imageContext.createCGImage(ciImage, from: ciImage.extent)
        
        let imagePath = outDir.appendingPathComponent(String(format: "%lld.jpg", timestampToInt(timestamp)))
        let options: NSDictionary = [kCGImageDestinationLossyCompressionQuality: 0.5]
        let myImageDest = CGImageDestinationCreateWithURL(imagePath as CFURL, kUTTypeJPEG, 1, nil)!
        CGImageDestinationAddImage(myImageDest, cgImage!, options)
        CGImageDestinationFinalize(myImageDest)
    }
    
    @available(iOS 14.0, *)
    func writeDepth(sceneDepth: ARDepthData, timestamp: TimeInterval) {
        let depthMap = sceneDepth.depthMap;
        CVPixelBufferLockBaseAddress(depthMap, CVPixelBufferLockFlags(rawValue: 0))
        let addr = CVPixelBufferGetBaseAddress(depthMap)
        let height = CVPixelBufferGetHeight(depthMap)
        let bpr = CVPixelBufferGetBytesPerRow(depthMap)
        let data = Data(bytes: addr!, count: (bpr*height))
        let fileName = String(format: "%lld.bin", timestampToInt(timestamp))
        let filePath = self.outDir.appendingPathComponent(fileName)
        do {
           try data.write(to: filePath)
        } catch {
            os_log("Cannot write depth map: %@", type:.error, error.localizedDescription)
        }
        CVPixelBufferUnlockBaseAddress(depthMap, CVPixelBufferLockFlags(rawValue: 0))
        
        if (sceneDepth.confidenceMap != nil) {
            let confidence = self.imageBufferToUIImage(buffer: sceneDepth.confidenceMap!)
            if let data = confidence.pngData() {
                let fileName = String(format: "%lld.confidence.png", timestampToInt(timestamp))
                let imagePath = self.outDir.appendingPathComponent(fileName)
                do {
                    try data.write(to: imagePath)
                } catch {
                    os_log("Cannot write confidence: %@", type:.error, error.localizedDescription)
                }
            }
        }
    }
}

class PoseWriter {
    var file: FileHandle!
    let filename = "poses.txt"
    let header = "# timestamp, status, tx, ty, tz, qx, qy, qz, qw, w, h, fx, fy, cx, cy, exposure\n"
    let template = "%lld, %@, %.6f, %.6f, %.6f, %.6f, %.6f, %.6f, %.6f, %u, %u, %.6f, %.6f, %.6f, %.6f, %lld\n"
    
    init?(outDir: URL) {
        let device = UIDevice.current  // add some info about the device used to record
        let device_str = String(format: "# %@ %@ %@ %@", device.name, device.systemName, device.systemVersion, device.model)
        let header_ = device_str + "\n" + header
        
        let fileURL = outDir.appendingPathComponent(filename)
        if (!FileManager.default.createFile(atPath: fileURL.path, contents: header_.data(using: String.Encoding.utf8), attributes: nil)) {
            os_log("Cannot create the pose file at %@", type:.error, fileURL.path)
            return nil
        }
        do {
            try file = FileHandle(forWritingTo: fileURL)
        } catch {
            os_log("Cannot create the pose file: %@", type:.error, error.localizedDescription)
            return nil
        }
        file.seekToEndOfFile()
    }
    
    func write(camera: ARCamera, timestamp: TimeInterval, state: String) {
        let tvec = camera.transform.columns.3
        let qvec = simd_quatf(camera.transform).vector
        let width = UInt32(camera.imageResolution.width)
        let height = UInt32(camera.imageResolution.height)
        let K = camera.intrinsics
        let poseData = String(
            format: template,
            timestampToInt(timestamp), state, tvec.x, tvec.y, tvec.z, qvec.x, qvec.y, qvec.z, qvec.w,
            width, height, K[0][0], K[1][1], K[2][0], K[2][1],
            timestampToInt(camera.exposureDuration))
        if let poseDataOut = poseData.data(using: .utf8) {
            file!.write(poseDataOut)
        } else {
            os_log("Failed to format to the pose string: %@", type: .fault, poseData)
        }
    }
    
    func finish() {
        file.closeFile()
        file = nil
    }
}

class AccelWriter {
    var file: FileHandle!
    var manager: CMMotionManager!
    let filename = "accelerometer.txt"
    let header = "# timestamp, ax, ay, az\n"
    let template = "%lld, %.6f, %.6f, %.6f\n"
    
    init?(outDir: URL, manager: CMMotionManager, freq: Double) {
        if !manager.isAccelerometerAvailable { return nil }
        manager.accelerometerUpdateInterval = 1.0 / freq
        self.manager = manager
        
        let fileURL = outDir.appendingPathComponent(filename)
        if (!FileManager.default.createFile(atPath: fileURL.path, contents: header.data(using: String.Encoding.utf8),
                                            attributes: nil)) {
            os_log("Cannot create the accelerometer file at %@", type:.error, fileURL.path)
            return nil
        }
        do {
            try file = FileHandle(forWritingTo: fileURL)
        } catch {
            os_log("Cannot create the accelerometer file: %@", type:.error, error.localizedDescription)
            return nil
        }
        file.seekToEndOfFile()
    }
    
    func start(queue: OperationQueue) {
        manager.startAccelerometerUpdates(to: queue, withHandler: { (inData, error) in
            if let data = inData {  // if valid
                let strData = String(
                    format: self.template,
                    timestampToInt(data.timestamp),
                    data.acceleration.x, data.acceleration.y, data.acceleration.z)
                if let outData = strData.data(using: .utf8) {
                    self.file!.write(outData)
                } else {
                    os_log("Failed to format to the accelerometer string: %@", type: .fault, strData)
                }
            }
        })
    }
    
    func finish() {
        manager.stopAccelerometerUpdates()
        file.closeFile()
        file = nil
    }
}

class GyroWriter {
    var file: FileHandle!
    var manager: CMMotionManager!
    let filename = "gyroscope.txt"
    let header = "# timestamp, rx, ry, rz\n"
    let template = "%lld, %.6f, %.6f, %.6f\n"
    
    init?(outDir: URL, manager: CMMotionManager, freq: Double) {
        if !manager.isGyroAvailable { return nil }
        manager.gyroUpdateInterval = 1.0 / freq
        self.manager = manager
        
        let fileURL = outDir.appendingPathComponent(filename)
        if (!FileManager.default.createFile(atPath: fileURL.path, contents: header.data(using: String.Encoding.utf8),
                                            attributes: nil)) {
            os_log("Cannot create the gyroscope file at %@", type:.error, fileURL.path)
            return nil
        }
        do {
            try file = FileHandle(forWritingTo: fileURL)
        } catch {
            os_log("Cannot create the gyroscope file: %@", type:.error, error.localizedDescription)
            return nil
        }
        file.seekToEndOfFile()
    }
    
    func start(queue: OperationQueue) {
        manager.startGyroUpdates(to: queue, withHandler: { (inData, error) in
            if let data = inData {  // if valid
                let strData = String(
                    format: self.template,
                    timestampToInt(data.timestamp),
                    data.rotationRate.x, data.rotationRate.y, data.rotationRate.z)
                if let outData = strData.data(using: .utf8) {
                    self.file!.write(outData)
                } else {
                    os_log("Failed to format to the gyroscope string: %@", type: .fault, strData)
                }
            }
        })
    }
    
    func finish() {
        manager.stopGyroUpdates()
        file.closeFile()
        file = nil
    }
}

class MagnetoWriter {
    var file: FileHandle!
    var manager: CMMotionManager!
    let filename = "magnetometer.txt"
    let header = "# timestamp, mx, my, mz\n"
    let template = "%lld, %.6f, %.6f, %.6f\n"
    
    init?(outDir: URL, manager: CMMotionManager, freq: Double) {
        if !manager.isMagnetometerAvailable { return nil }
        manager.magnetometerUpdateInterval = 1.0 / freq
        self.manager = manager
        
        let fileURL = outDir.appendingPathComponent(filename)
        if (!FileManager.default.createFile(atPath: fileURL.path, contents: header.data(using: String.Encoding.utf8),
                                            attributes: nil)) {
            os_log("Cannot create the magnetometer file at %@", type:.error, fileURL.path)
            return nil
        }
        do {
            try file = FileHandle(forWritingTo: fileURL)
        } catch {
            os_log("Cannot create the magnetometer file: %@", type:.error, error.localizedDescription)
            return nil
        }
        file.seekToEndOfFile()
    }
    
    func start(queue: OperationQueue) {
        manager.startMagnetometerUpdates(to: queue, withHandler: { (inData, error) in
            if let data = inData {  // if valid
                let strData = String(
                    format: self.template,
                    timestampToInt(data.timestamp),
                    data.magneticField.x, data.magneticField.y, data.magneticField.z)
                if let outData = strData.data(using: .utf8) {
                    self.file!.write(outData)
                } else {
                    os_log("Failed to format to the magnetometer string: %@", type: .fault, strData)
                }
            }
        })
    }
    
    func finish() {
        manager.stopMagnetometerUpdates()
        file.closeFile()
        file = nil
    }
}

class FusedMotionWriter {
    var file: FileHandle!
    var manager: CMMotionManager!
    let filename = "fused_imu.txt"
    let header = "# timestamp, ax, ay, az, rx, ry, rz, mx, my, mz, gx, gy, gz, heading\n"
    let template = "%lld, " + Array(repeating: "%.6f", count: 13).joined(separator: ", ") + "\n"
    
    init?(outDir: URL, manager: CMMotionManager, freq: Double) {
        if !manager.isDeviceMotionAvailable { return nil }
        manager.deviceMotionUpdateInterval = 1.0 / freq
        self.manager = manager
        
        let fileURL = outDir.appendingPathComponent(filename)
        if (!FileManager.default.createFile(atPath: fileURL.path, contents: header.data(using: String.Encoding.utf8),
                                            attributes: nil)) {
            os_log("Cannot create the fused motion file at %@", type:.error, fileURL.path)
            return nil
        }
        do {
            try file = FileHandle(forWritingTo: fileURL)
        } catch {
            os_log("Cannot create the fused motion file: %@", type:.error, error.localizedDescription)
            return nil
        }
        file.seekToEndOfFile()
    }
    
    func start(queue: OperationQueue) {
        manager.startDeviceMotionUpdates(using: CMAttitudeReferenceFrame.xTrueNorthZVertical, to: queue, withHandler: { (inData, error) in
            if let data = inData {  // if valid
                let strData = String(
                    format: self.template,
                    timestampToInt(data.timestamp),
                    data.userAcceleration.x, data.userAcceleration.y, data.userAcceleration.z,
                    data.rotationRate.x, data.rotationRate.y, data.rotationRate.z,
                    data.magneticField.field.x, data.magneticField.field.y, data.magneticField.field.z,
                    data.gravity.x, data.gravity.y, data.gravity.z,
                    data.heading)
                if let outData = strData.data(using: .utf8) {
                    self.file!.write(outData)
                } else {
                    os_log("Failed to format to the fused motion string: %@", type: .fault, strData)
                }
            }
        })
    }
    
    func finish() {
        manager.stopDeviceMotionUpdates()
        file.closeFile()
        file = nil
    }
}

class MotionWriter {
    var accelWriter: AccelWriter!
    var gyroWriter: GyroWriter!
    var magnetoWriter: MagnetoWriter!
    var fusedWriter: FusedMotionWriter!
    var manager: CMMotionManager
    var queue: OperationQueue!
    
    init?(outDir: URL, manager: CMMotionManager, freq: Double) {
        guard let accelWriter = AccelWriter(outDir: outDir, manager: manager, freq: freq) else {return nil}
        self.accelWriter = accelWriter
        guard let gyroWriter = GyroWriter(outDir: outDir, manager: manager, freq: freq) else {return nil}
        self.gyroWriter = gyroWriter
        guard let magnetoWriter = MagnetoWriter(outDir: outDir, manager: manager, freq: freq) else {return nil}
        self.magnetoWriter = magnetoWriter
        guard let fusedWriter = FusedMotionWriter(outDir: outDir, manager: manager, freq: freq) else {return nil}
        self.fusedWriter = fusedWriter
        self.manager = manager
    }
    
    func start() {
        queue = OperationQueue()
        queue.name = "IMU queue"
        queue.maxConcurrentOperationCount = 1
        
        accelWriter!.start(queue: queue)
        gyroWriter!.start(queue: queue)
        magnetoWriter!.start(queue: queue)
        fusedWriter!.start(queue: queue)
    }
    
    func finish() {
        manager.stopAccelerometerUpdates()
        manager.stopGyroUpdates()
        manager.stopMagnetometerUpdates()
        manager.stopDeviceMotionUpdates()
        queue.waitUntilAllOperationsAreFinished()  // use a DispatchGroup if this blocks for too long
        accelWriter.finish()
        gyroWriter.finish()
        magnetoWriter.finish()
        fusedWriter.finish()
    }
}

class BluetoothWriter {
    var file: FileHandle!
    let filename = "bluetooth.txt"
    let header = "# timestamp, name, uuid, rssi\n"
    let template = "%lld, %@, %@, %@\n"
    
    init?(outDir: URL) {
        let fileURL = outDir.appendingPathComponent(filename)
        if (!FileManager.default.createFile(atPath: fileURL.path, contents: header.data(using: String.Encoding.utf8), attributes: nil)) {
            os_log("Cannot create the bluetooth file at %@", type:.error, fileURL.path)
            return nil
        }
        do {
            try file = FileHandle(forWritingTo: fileURL)
        } catch {
            os_log("Cannot create the bluetooth file: %@", type:.error, error.localizedDescription)
            return nil
        }
        file.seekToEndOfFile()
    }
    
    func write(peripheral: CBPeripheral, rssi: NSNumber) {
        let name = peripheral.name ?? "unkown"
        let strData = String(
            format: template,
            timestampToInt(ProcessInfo.processInfo.systemUptime),
            name.replacingOccurrences(of: ",", with: ""),
            peripheral.identifier.uuidString,
            rssi)
        if let outData = strData.data(using: .utf8) {
            file!.write(outData)
        } else {
            os_log("Failed to format to the bluetooth string: %@", type: .fault, strData)
        }
    }
    
    func finish() {
        file.closeFile()
        file = nil
    }
}

class LocationWriter {
    var file: FileHandle!
    let filename = "location.txt"
    let header = "# timestamp, lat, long, z, sigma_xy, sigma_z\n"
    let template = "%lld, %.6f, %.6f, %.6f, %.6f, %.6f\n"

    init?(outDir: URL) {
        let fileURL = outDir.appendingPathComponent(filename)
        if (!FileManager.default.createFile(atPath: fileURL.path, contents: header.data(using: String.Encoding.utf8), attributes: nil)) {
            os_log("Cannot create the location file at %@", type:.error, fileURL.path)
            return nil
        }
        do {
            try file = FileHandle(forWritingTo: fileURL)
        } catch {
            os_log("Cannot create the location file: %@", type:.error, error.localizedDescription)
            return nil
        }
        file.seekToEndOfFile()
    }

    func write(location: CLLocation) {
        let bootDate = Date() - ProcessInfo.processInfo.systemUptime
        let strData = String(
            format: template,
            timestampToInt(location.timestamp.timeIntervalSince(bootDate)),
            location.coordinate.latitude,
            location.coordinate.longitude,
            location.altitude,
            location.horizontalAccuracy,
            location.verticalAccuracy)
        if let outData = strData.data(using: .utf8) {
            file!.write(outData)
        } else {
            os_log("Failed to format to the location string: %@", type: .fault, strData)
        }
    }
    
    func write_multiple(locations: [CLLocation]) {
        for location in locations {
            write(location: location)
        }
    }

    func finish() {
        file.closeFile()
        file = nil
    }
}

