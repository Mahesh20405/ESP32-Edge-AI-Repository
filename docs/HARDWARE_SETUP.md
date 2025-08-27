# Hardware Setup Guide

## ESP32 Development Board Setup

### Required Components
- ESP32 development board (ESP32-WROOM-32 or ESP32-DevKitC)
- Breadboard
- Jumper wires (male-to-male, male-to-female)
- LED indicators (3mm or 5mm)
- Resistors (220Ω for LEDs, 10kΩ for pull-up)
- Push buttons (tactile switches)
- USB cable (micro-USB or USB-C depending on board)

### Optional Sensors (for extended functionality)
- MPU6050 (6-axis accelerometer/gyroscope)
- BME280 (temperature, humidity, pressure)
- Light sensor (LDR or photodiode)

## Pin Configuration

### ESP32 GPIO Pin Assignments
```
GPIO Pin    | Function              | Connection
-----------|-----------------------|------------------
GPIO 2     | Built-in LED          | Status indicator
GPIO 4     | Button input          | Inference trigger
GPIO 16    | External LED (Red)    | Error indicator  
GPIO 17    | External LED (Green)  | Success indicator
GPIO 21    | I2C SDA              | Sensor data line
GPIO 22    | I2C SCL              | Sensor clock line
GPIO 34    | Analog input          | Sensor reading
3.3V       | Power supply          | Sensor power
GND        | Ground                | Common ground
```

### Power Requirements
- **Supply Voltage**: 3.3V (regulated)
- **Operating Current**: 80-240mA (depending on WiFi usage)
- **Deep Sleep Current**: 5-10µA
- **Flash Memory**: 4MB (minimum for TensorFlow Lite Micro)
- **RAM**: 520KB (320KB available for applications)

## Circuit Connections

### Basic Inference Setup
```
ESP32 Board                    Components
-----------                    ----------
3.3V    ──────────────────────  VCC (Sensors)
GND     ──────────────────────  GND (Common)
GPIO 2  ──────────────────────  Built-in LED
GPIO 4  ────[10kΩ]──── 3.3V     
        └── [Button] ── GND     Inference Trigger
GPIO 16 ────[220Ω]──── LED ──── GND (Red - Error)
GPIO 17 ────[220Ω]──── LED ──── GND (Green - Success)
```

### I2C Sensor Interface (Optional)
```
ESP32          Sensor (MPU6050/BME280)
-----          ------------------------
3.3V    ────── VCC
GND     ────── GND  
GPIO 21 ────── SDA
GPIO 22 ────── SCL
```

## Circuit Diagram Generation

### Method 1: Using Fritzing
1. **Download Fritzing**: https://fritzing.org/
2. **Import ESP32 part**: Use ESP32-DevKitC from parts library
3. **Add components**: LEDs, resistors, buttons from core parts
4. **Create connections** following pin assignments above
5. **Export as PNG** for documentation

### Method 2: ASCII Circuit Diagram
```
                    ESP32-DevKitC
                   ┌─────────────┐
    3.3V ──────────┤ 3.3V    GND ├─────────── GND
                   │             │
    Button ────────┤ GPIO4  GPIO2├─────────── Built-in LED
    (with 10kΩ)    │             │
                   │      GPIO16 ├──[220Ω]── Red LED ── GND
    Sensor SDA ────┤ GPIO21      │
                   │      GPIO17 ├──[220Ω]── Green LED ── GND
    Sensor SCL ────┤ GPIO22      │
                   └─────────────┘
```

### Method 3: Using Python schemdraw
```python
import schemdraw
import schemdraw.elements as elm

# Create circuit diagram
d = schemdraw.Drawing()

# ESP32 microcontroller
esp32 = d.add(elm.Ic(pins=[
    elm.IcPin(name='3V3', side='left'),
    elm.IcPin(name='GND', side='left'),
    elm.IcPin(name='GPIO2', side='right'),
    elm.IcPin(name='GPIO4', side='right'),
    elm.IcPin(name='GPIO16', side='right'),
    elm.IcPin(name='GPIO17', side='right')
], edgepadx=1.5, edgepady=.5, label='ESP32'))

# Power connections
d.add(elm.Line().left().at(esp32.GPIO4).length(1))
d.add(elm.Button().label('Inference\nTrigger'))

# LED indicators
d.add(elm.Line().right().at(esp32.GPIO16).length(1))
d.add(elm.Resistor().label('220Ω'))
d.add(elm.Led().label('Error LED'))

d.save('docs/figures/circuit_diagram.png')
```

## ESP32 Programming Setup

### Arduino IDE Configuration
1. **Install ESP32 Board Package**:
   - Go to File → Preferences
   - Add to Additional Board Manager URLs:
     `https://dl.espressif.com/dl/package_esp32_index.json`
   - Install ESP32 by Espressif Systems

2. **Install Required Libraries**:
   ```
   TensorFlowLite_ESP32
   ArduinoJson
   WiFi (built-in)
   ```

### PlatformIO Configuration
Create `platformio.ini`:
```ini
[env:esp32dev]
platform = espressif32
board = esp32dev
framework = arduino
monitor_speed = 115200
lib_deps = 
    tensorflow/TensorFlowLite_ESP32@^2.4.0
    bblanchon/ArduinoJson@^6.19.4
build_flags = 
    -DCORE_DEBUG_LEVEL=3
    -DBOARD_HAS_PSRAM
```

## Deployment Checklist

### Pre-deployment Tests
- [ ] Model fits in flash memory (< 4MB)
- [ ] Inference time < 50ms target
- [ ] Memory usage < 32KB RAM
- [ ] Power consumption acceptable
- [ ] All GPIO connections verified
- [ ] Serial communication working

### Performance Verification
```cpp
// ESP32 timing test code
unsigned long start_time = micros();
interpreter->Invoke();
unsigned long inference_time = micros() - start_time;

Serial.print("Inference time: ");
Serial.print(inference_time / 1000.0);
Serial.println(" ms");
```

### Troubleshooting Common Issues

| Issue | Solution |
|-------|----------|
| Model too large | Apply more aggressive quantization |
| Slow inference | Reduce model complexity or use pruning |
| Memory errors | Optimize tensor allocations |
| GPIO not working | Check pin assignments and voltage levels |
| WiFi interference | Use deep sleep between inferences |

## Safety Considerations

### Electrical Safety
- Always disconnect power when wiring
- Use appropriate current-limiting resistors
- Avoid short circuits on power rails
- Check polarity of LEDs and sensors

### EMC Considerations
- Keep wire lengths short for high-frequency signals
- Use bypass capacitors (0.1µF) near power pins
- Avoid running wires parallel to power lines
- Shield sensitive analog inputs if needed

## Next Steps

1. **Flash Firmware**: Upload TensorFlow Lite Micro sketch
2. **Test Inference**: Verify model predictions
3. **Optimize Performance**: Fine-tune for target metrics
4. **Deploy**: Install in final application environment

---

# Circuit Diagram Details

## Schematic Symbol Reference

### ESP32-WROOM-32 Pinout
```
                     ESP32-WROOM-32
                   ┌─────────────────┐
            3V3 ──┤1              38├── GND
             EN ──┤2              37├── GPIO23
          GPIO36 ──┤3              36├── GPIO22 (SCL)
          GPIO39 ──┤4              35├── GPIO21 (SDA)  
          GPIO34 ──┤5              34├── GPIO19
          GPIO35 ──┤6              33├── GPIO18
          GPIO32 ──┤7              32├── GPIO5
          GPIO33 ──┤8              31├── GPIO17
          GPIO25 ──┤9              30├── GPIO16
          GPIO26 ──┤10             29├── GPIO4
          GPIO27 ──┤11             28├── GPIO0
          GPIO14 ──┤12             27├── GPIO2
          GPIO12 ──┤13             26├── GPIO15
            GND ──┤14             25├── GPIO8
          GPIO13 ──┤15             24├── GPIO7
           GPIO9 ──┤16             23├── GPIO6
          GPIO10 ──┤17             22├── GPIO1 (TX)
          GPIO11 ──┤18             21├── GPIO3 (RX)
            VIN ──┤19             20├── GND
                   └─────────────────┘
```

### Component Values
- **LED Current Limiting Resistors**: 220Ω (for 3.3V supply)
- **Button Pull-up Resistors**: 10kΩ
- **I2C Pull-up Resistors**: 4.7kΩ (if not internal)
- **Bypass Capacitors**: 0.1µF ceramic (close to power pins)

This hardware setup provides a complete foundation for deploying your neural network model on ESP32 with proper monitoring and debugging capabilities.