
# Nhận diện văn bản từ ảnh

# Giới thiệu
Ứng dụng này cho phép bạn chọn một hoặc nhiều hình ảnh và phân tích văn bản trong hình ảnh đó bằng cách sử dụng các mô hình học sâu. Nó kết hợp giữa mô hình phát hiện văn bản YOLO và mô hình nhận dạng văn bản CRNN để trích xuất và nhận dạng văn bản từ ảnh.

# Yêu cầu
- Python 3.x
- Các thư viện Python: matplotlib, PyQt5, torchvision, nltk, torch, timm, ultralytics
Cài đặt các thư viện bằng lệnh sau:

              pip install matplotlib PyQt5 torchvision nltk torch timm ultralytics

# Cách sử dụng
1.Chọn ảnh: Nhấn nút "Chọn ảnh" để chọn một hoặc nhiều tập tin hình ảnh (định dạng jpg, png, jpeg) từ hệ thống tệp của bạn.

2.Phân tích văn bản: Sau khi chọn ảnh, ứng dụng sẽ tự động phân tích văn bản trong ảnh đó và hiển thị kết quả lên giao diện người dùng. Kết quả bao gồm văn bản được nhận dạng từ ảnh và hình ảnh kết quả với các hộp giới hạn và văn bản nhận dạng được vẽ lên.

# Hướng dẫn cài đặt
1.Clone repository này về máy của bạn:
        
        git clone https://github.com/nguyenhoanghiep478/RecognizeImage.git
2.Chuyển đến thư mục của dự án:

        cd RecognizeImage
3.Tải về 2 file :
    
       https://drive.google.com/drive/folders/1TS5eZLMgsqjF_KScDNzz0XuC3aYADjDc
4.Chạy ứng dụng:
        
        python main.py

# Tài liệu và Mô hình
-  Mô hình YOLOv5 cho phát hiện văn bản: YOLOv5 repository
- Mô hình CRNN cho nhận dạng văn bản: CRNN repository
# Tác giả 
- Tên : Nguyễn Hoàng Hiệp, Nguyễn Ngọc Huy
