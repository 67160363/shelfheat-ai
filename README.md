# ShelfHeat AI

ระบบ AI วิเคราะห์ภาพจากกล้องวงจรปิด (CCTV) เพื่อสร้าง **Heatmap** และ **Store Analytics** ของพฤติกรรมลูกค้าในร้านค้าปลีก พร้อมเบลอใบหน้าอัตโนมัติตามหลัก **PDPA (Privacy by Design)**

![Dashboard Screenshot](docs/dashboard.png) <!-- TODO: ใส่ภาพหน้าจอ Dashboard -->

## ฟีเจอร์หลัก
- ตรวจจับและติดตามบุคคลแบบเรียลไทม์ (YOLOv8 + ByteTrack)
- สร้าง Heatmap แสดงโซนที่ลูกค้าเดินผ่าน/หยุดดูสินค้ามากที่สุด
- Store Analytics: จำนวนคน, ระยะเวลาอยู่ในร้าน และสถิติต่าง ๆ ผ่าน API
- เบลอใบหน้า/ศีรษะอัตโนมัติเพื่อปกป้องข้อมูลส่วนบุคคล (PDPA)
- Dashboard บนเว็บ พร้อมสตรีมวิดีโอสด และปุ่มเปิด/ปิด Privacy และ Heatmap

## Tech Stack
- Python, FastAPI, Uvicorn
- Ultralytics YOLOv8, ByteTrack
- OpenCV, NumPy
- HTML / CSS / JavaScript (Dashboard)

## การติดตั้งและใช้งาน
```bash
pip install -r requirements.txt
```
1. นำไฟล์วิดีโอ CCTV ไปไว้ที่ `videos/sample.mp4` (หรือกำหนด path เองผ่านตัวแปร `VIDEO_PATH`)
2. รันเซิร์ฟเวอร์
   ```bash
   uvicorn main:app --reload
   ```
3. เปิดเบราว์เซอร์ไปที่ http://127.0.0.1:8000

> โมเดล YOLOv8 (`.pt`) จะถูกดาวน์โหลดอัตโนมัติในการรันครั้งแรก

## ผู้จัดทำ
โภคิน เกตุมะยูร
