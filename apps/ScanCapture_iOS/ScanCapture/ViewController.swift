//
//  ViewController.swift
//  ScanCapture
//
//  Created by Paul-Edouard Sarlin on 26.04.21.
//

import UIKit
import SceneKit
import ARKit
import os.log
import Accelerate
import CoreMotion
import CoreBluetooth
import CoreLocation

class ViewController: UIViewController, ARSCNViewDelegate, ARSessionDelegate, CBCentralManagerDelegate, CLLocationManagerDelegate {
    
    // cellphone screen UI outlet objects
    @IBOutlet weak var startStopButton: UIButton!
    @IBOutlet weak var timeLabel: UILabel!
    @IBOutlet weak var trackingStatusLabel: UILabel!
    @IBOutlet weak var mappingStatusLabel: UILabel!
    @IBOutlet weak var frameCounterLabel: UILabel!
    @IBOutlet weak var fileSizeLabel: UILabel!
    @IBOutlet weak var fpsLabel: UILabel!
    @IBOutlet weak var fpsStepper: UIStepper!
    @IBOutlet weak var timeWriteLabel: UILabel!
    
    @IBOutlet var sceneView: ARSCNView!
    
    var isRecording: Bool = false
    let queue: DispatchQueue = DispatchQueue(label: "com.scantoolscapture", attributes: .concurrent)
    var writerQueue: OperationQueue!
    var hasDepth: Bool = false
    
    var frameDrop: UInt = 6  // how many frames to skip at 60Hz
    var arFrameCounter: UInt = 0  // total number of ARFrames at 60Hz
    var captureFrameCounter: UInt = 0  // number of subsampled frames
    
    var outDirURL: URL!
    let imageDirName = "images"
    let depthDirName = "depth"
    var imageWriter: ImageStreamer!
    var poseWriter: PoseWriter!
    
    let captureDepth: Bool = true
    var depthWriter: ImageWriter!
    
    let captureIMU: Bool = true
    let imuFreq: Double = 100.0
    var motionManager: CMMotionManager!
    var motionWriter: MotionWriter!
    
    let captureBT: Bool = true
    var btManager: CBCentralManager!
    var btTimer: Timer!
    var btWriter: BluetoothWriter!
    let btQueue: DispatchQueue = DispatchQueue(label: "com.scantoolscapture.bluetooth")
    
    let captureLocation: Bool = true
    var locationManager: CLLocationManager!
    var locationWriter: LocationWriter!
    
    // UI
    var diskCapacity: String = "?"
    var startTime: Date!
    var recordingTimer: Timer!
    var previousPosition: SCNVector3?
    var timeWriteText: String = "? ms"
    
    override func viewDidLoad() {
        super.viewDidLoad()
        
        self.sceneView.debugOptions = [ARSCNDebugOptions.showFeaturePoints, ARSCNDebugOptions.showWorldOrigin]

        sceneView.delegate = self
        sceneView.showsStatistics = true
        sceneView.session.delegate = self
        
        updateDiskCapacity()
        initializeUI()
        startStopButton.setTitle("Start", for: .normal)

        writerQueue = OperationQueue()
        writerQueue.maxConcurrentOperationCount = 1
        
        if captureIMU {
            motionManager = CMMotionManager()
            if !motionManager.isDeviceMotionAvailable {os_log("Fused device motion not available.")}
            if !motionManager.isGyroAvailable {os_log("Gyroscope not available.")}
            if !motionManager.isAccelerometerAvailable {os_log("Accelerometer not available.")}
            if !motionManager.isMagnetometerAvailable {os_log("Magnetometer not available.")}
        }
        if captureBT {
            btManager = CBCentralManager(delegate: self, queue: btQueue)
        }
        if captureLocation {
            locationManager = CLLocationManager()
            locationManager!.delegate = self
            locationManager!.desiredAccuracy = kCLLocationAccuracyBest
            locationManager!.distanceFilter = kCLDistanceFilterNone
            locationManager!.requestWhenInUseAuthorization()
        }
    }
    
    override func viewWillAppear(_ animated: Bool) {
        super.viewWillAppear(animated)
    
        let configuration = ARWorldTrackingConfiguration()
        configuration.worldAlignment = ARConfiguration.WorldAlignment.gravity
        if #available(iOS 14.0, *) {
            if captureDepth && ARWorldTrackingConfiguration.supportsFrameSemantics([.sceneDepth]) {
                configuration.frameSemantics = [.sceneDepth]
                self.hasDepth = true
                os_log("Will also save depth data.")
            }
        }
        
        sceneView.session.run(configuration)
    }
    
    override func viewWillDisappear(_ animated: Bool) {
        super.viewWillDisappear(animated)
        
        sceneView.session.pause()
    }

    @IBAction func startStopButtonPressed(_ sender: UIButton) {
        if (self.isRecording == false) {
            os_log("Starting a new recording.")
            queue.async {
                if (self.createFiles()) {
                    DispatchQueue.main.async {
                        // reset timer
                        self.startTime = Date()
                        self.updateTime()
                        self.recordingTimer = Timer.scheduledTimer(
                            timeInterval: 1.0, target: self, selector: #selector(self.updateTime),
                            userInfo: nil, repeats: true)
                        self.arFrameCounter = 0
                        self.captureFrameCounter = 0
                        self.initializeUI()
                        self.toggleRecording(val: true)
                        if self.btWriter != nil {
                            self.btTimer = Timer.scheduledTimer(
                                timeInterval: 1.0, target: self, selector: #selector(self.refreshBluetooth),
                                userInfo: nil, repeats: true)
                        }
                        if self.locationWriter != nil {
                            self.locationManager!.startUpdatingLocation()
                        }
                    }
                    self.motionWriter?.start()
                } else {
                    self.showError(msg: "Failed to create the recording directory or files.")
                    return
                }
            }
        } else {
            os_log("Stopping the recording.")
            self.toggleRecording(val: false)
            if recordingTimer?.isValid == true {
                recordingTimer.invalidate()
            }
            if self.captureBT {
                if btTimer?.isValid == true {
                    btTimer.invalidate()
                }
                self.stopBluetoooth()
            }
            if self.captureLocation {
                self.locationManager!.stopUpdatingLocation()
            }
            self.writerQueue.addBarrierBlock({
                os_log("Finishing all writers")
                self.poseWriter.finish()
                self.imageWriter.finish()
                self.motionWriter?.finish()
                self.btWriter?.finish()
                self.locationWriter?.finish()
                os_log("Opening the capture directory.")
                var sharedURL = URLComponents(url: self.outDirURL!, resolvingAgainstBaseURL: false)!
                sharedURL.scheme = "shareddocuments"  // scheme of the Files app
                DispatchQueue.main.async {
                    UIApplication.shared.open(sharedURL.url!)
                }
            })
        }
    }
    
    func session(_ session: ARSession, didFailWithError error: Error) {
        os_log("AR session failed: %@", type:.error, error.localizedDescription)
    }
    
    func session(_ session: ARSession, didUpdate frame: ARFrame) {
        let mappingStatus = frame.worldMappingStatus
        let trackingState = frame.camera.trackingState

        if (self.arFrameCounter % 6) == 0 {  // only update the UI every 100ms
            DispatchQueue.main.async { [arFrameCounter = self.arFrameCounter] in
                self.trackingStatusLabel.text = trackingState.toString()
                self.mappingStatusLabel.text = mappingStatus.toString()
                self.timeWriteLabel.text = self.timeWriteText + String(format:" / %d", self.writerQueue.operationCount)
                if self.isRecording {
                    if (arFrameCounter % 30) == 0 {  // every half second
                        self.drawCamera(camera: frame.camera)
                        if (arFrameCounter % 1200) == 0 {  // every 20 seconds
                            self.updateSize()
                        }
                    }
                }
            }
        }

        if (self.isRecording) {
            let timestamp = frame.timestamp
            let camera = frame.camera
            let imageBuffer = frame.capturedImage

            if (arFrameCounter % frameDrop == 0) {
                captureFrameCounter += 1
                frameCounterLabel.text = String(format: "%u", captureFrameCounter)
                self.writerQueue.addOperation({
                    let enter = ProcessInfo.processInfo.systemUptime
                    self.poseWriter.write(camera: camera, timestamp: timestamp, state: trackingState.toString())
                    self.imageWriter.write(buffer: imageBuffer, timestamp: timestamp)
                    if #available(iOS 14.0, *) {
                        if (self.hasDepth && (frame.sceneDepth != nil)) {
                            self.depthWriter.writeDepth(sceneDepth: frame.sceneDepth!, timestamp: timestamp)
                        }
                    }
                    self.timeWriteText = String(format: "%.1f ms", (ProcessInfo.processInfo.systemUptime - enter)*1000)
                })
            }
            arFrameCounter += 1
        }
    }
    
    private func drawCamera(camera: ARCamera) {
        let tvec = camera.transform.columns.3
        let position = SCNVector3Make(tvec.x, tvec.y, tvec.z)
        
        if previousPosition != nil {
            let dist = (position - previousPosition!).length()
            if dist < 1.0 {  // every meter
                return
            }
        }
        
        let node = SCNNode(geometry: SCNSphere(radius: 0.005))
        node.geometry?.firstMaterial?.diffuse.contents = UIColor.red
        node.simdPosition = simd_make_float3(tvec)
        sceneView.scene.rootNode.addChildNode(node)
        
        if previousPosition != nil {
            let indices: [Int32] = [0, 1]
            let source = SCNGeometrySource(vertices: [previousPosition!, position])
            let element = SCNGeometryElement(indices: indices, primitiveType: .line)
            let line = SCNGeometry(sources: [source], elements: [element])
            let lineNode = SCNNode(geometry: line)
            lineNode.geometry?.firstMaterial?.diffuse.contents = UIColor.white
            sceneView.scene.rootNode.addChildNode(lineNode)
        }
        previousPosition = position
    }
    
    private func toggleRecording(val: Bool) {
        self.isRecording = val
        if val {
            self.startStopButton.setTitle("Stop", for: .normal)
            self.fpsStepper.isEnabled = false
            // prevent screen lock
            UIApplication.shared.isIdleTimerDisabled = true
        } else {
            self.startStopButton.setTitle("Start", for: .normal)
            self.fpsStepper.isEnabled = true
            // re-allow screen lock
            UIApplication.shared.isIdleTimerDisabled = false
        }
    }
    
    @IBAction func fpsStepperChanged(_ sender: UIStepper) {
        frameDrop = 61 - UInt(sender.value)
        fpsLabel.text = String(format: "%.1f FPS", 60/Double(frameDrop))
    }
    
    private func initializeUI() {
        timeLabel.text = "Ready"
        frameCounterLabel.text = "0"
        fileSizeLabel.text = String(format: "? / %@", self.diskCapacity)
        fpsLabel.text = String(format: "%.1f FPS", 60/Double(frameDrop))
        fpsStepper.value = Double(61 - frameDrop)
        timeWriteLabel.text = timeWriteText
        
        sceneView.scene.rootNode.enumerateChildNodes { (node, stop) in
                node.removeFromParentNode()
        }
    }
    
    @objc private func updateTime() {
        var elapsed = Int64(round(Date().timeIntervalSince(self.startTime)))
        let hours: Int64 = elapsed / 3600
        elapsed = elapsed % 3600
        let mins: Int64 = elapsed / 60
        let secs: Int64 = elapsed % 60
        self.timeLabel.text = String(format: "%02d:%02d:%02d", hours, mins, secs)
    }
    
    private func updateSize() {
        var str: String = "?"
        if let size = try? self.outDirURL.sizeOnDisk(){
             str = size
        }
        self.fileSizeLabel.text = String(format: "%@ / %@", str, self.diskCapacity)
    }
    
    private func showError(msg: String) {
        DispatchQueue.main.async {
            let fileAlert = UIAlertController(title: "Error", message: msg, preferredStyle: .alert)
            fileAlert.addAction(UIAlertAction(title: "OK", style: .cancel, handler: nil))
            self.present(fileAlert, animated: true, completion: nil)
        }
    }
    
    private func createFiles() -> Bool {
        // Create the output directory
        let recDirURL = getRecDir()
        let dateFormatter = DateFormatter()
        dateFormatter.dateFormat = "yyyy-MM-dd_HH.mm.ss"
        let date = dateFormatter.string(from: Date())
        outDirURL = recDirURL.appendingPathComponent(date)
        do {
            try FileManager.default.createDirectory(at: outDirURL, withIntermediateDirectories: true, attributes: nil)
        } catch {
            os_log("Cannot create the output directory: %@", type:.error, error.localizedDescription)
            return false
        }
        
        // Create the pose file
        guard let poseWriter = PoseWriter(outDir: outDirURL) else {return false}
        self.poseWriter = poseWriter
        
        self.imageWriter = ImageStreamer(outDir: outDirURL)
        
        // Create the depth folder
        if (hasDepth) {
            let depthDirURL = outDirURL.appendingPathComponent(depthDirName)
            guard let depthWriter = ImageWriter(outDir: depthDirURL) else {return false}
            self.depthWriter = depthWriter
        }
        
        if captureIMU && (motionManager != nil){
            guard let motionWriter = MotionWriter(
                    outDir: outDirURL, manager: motionManager, freq: imuFreq) else {return false}
            self.motionWriter = motionWriter
        } else {
            os_log("Will not record IMU data.")
        }
        
        if captureBT && (self.btManager?.state == .poweredOn) {
            guard let btWriter = BluetoothWriter(outDir: outDirURL) else {return false}
            self.btWriter = btWriter
        } else {
            os_log("Will not record Bluetooth data.")
        }
        
        if captureLocation {
            let status: CLAuthorizationStatus
            if #available(iOS 14.0, *) {
                status = locationManager!.authorizationStatus
            } else {
                status = CLLocationManager.authorizationStatus()
            }
            if [CLAuthorizationStatus.authorizedAlways, CLAuthorizationStatus.authorizedWhenInUse].contains(status) {
                os_log("Location recording is enabled.")
                guard let locationWriter = LocationWriter(outDir: outDirURL) else {return false}
                self.locationWriter = locationWriter
            } else {
                os_log("Will not record Location data because it was not approved.")
            }
        }
        
        updateDiskCapacity()
        return true
    }
    
    private func updateDiskCapacity() {
        do {
            let capacityValues = try self.getRecDir().resourceValues(forKeys: [.volumeAvailableCapacityForImportantUsageKey])
            if let capacityBytes = capacityValues.volumeAvailableCapacityForImportantUsage {
                let limit = 100
                if capacityBytes > (limit*1024*1024*1024) {
                    self.diskCapacity = String(format: "%d+ GB", limit)
                } else {
                    self.diskCapacity = ByteCountFormatter.string(fromByteCount: capacityBytes, countStyle: .file)
                }
            }
        } catch {
        }
    }
    
    private func getRecDir() -> URL {
        return try! FileManager.default.url(for: .documentDirectory, in: .userDomainMask, appropriateFor: nil, create: true)
    }
    
    // Bluetooth methods
    internal func centralManagerDidUpdateState(_ central: CBCentralManager) {
        switch central.state {
            case .poweredOn:
                os_log("Bluetooth is enabled.")
            case .poweredOff:
                self.showError(msg: "Bluetooth is off - please turn it on.")
            case .unauthorized:
                self.showError(msg: "App does not have Bluetooth auth.")
            case .unsupported:
                self.showError(msg: "Device does not support Bluetooth.")
            case .unknown:
                os_log("Unknown Bluetooth error.")
            case .resetting:
                break
            @unknown default:
                break
        }
    }
    
    @objc private func refreshBluetooth() {
        stopBluetoooth()
        startBluetoooth()
    }
    
    private func startBluetoooth() {
        btManager.scanForPeripherals(
            withServices: nil, options: [CBCentralManagerScanOptionAllowDuplicatesKey : false])
    }
    
    private func stopBluetoooth() {
        if self.btManager?.state == .poweredOn {
            btManager.stopScan()
        }
    }
    
    func centralManager(_ central: CBCentralManager, didDiscover peripheral: CBPeripheral, advertisementData: [String : Any], rssi: NSNumber) {
        btWriter.write(peripheral: peripheral, rssi: rssi)
    }
    
    // Location methods
    func locationManager(_ manager: CLLocationManager, didFailWithError error: Error) {
        let clErr = error as! CLError
        switch clErr.code {
            case .denied:
                self.showError(msg: "Location service has been denied - please approve it.")
                self.locationManager!.stopUpdatingLocation()
                break
            case .headingFailure:
                os_log("Location service: Couldn't find the heading.")
                break
            default:
                break
        }
    }
    
    func locationManager(_ manager: CLLocationManager,  didUpdateLocations locations: [CLLocation]) {
        if locationWriter != nil {
            self.writerQueue.addOperation({
                self.locationWriter?.write_multiple(locations: locations)
            })
        }
    }
}
