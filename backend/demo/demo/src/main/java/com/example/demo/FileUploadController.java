package com.example.demo;

import org.springframework.http.ResponseEntity;
import org.springframework.web.bind.annotation.*;
import org.springframework.web.multipart.MultipartFile;

@RestController
@RequestMapping("/api") // 定义 API 的基础路径
public class FileUploadController {

    @PostMapping("/upload") // 定义 POST 请求路径为 /api/upload
    public ResponseEntity<String> handleFileUpload(@RequestParam("file") MultipartFile file) {
        try {
            String fileName = file.getOriginalFilename();
            System.out.println("上传的文件名：" + fileName);

            return ResponseEntity.ok("文件上传成功：" + fileName);
        } catch (Exception e) {
            e.printStackTrace();
            return ResponseEntity.status(500).body("文件上传失败！");
        }
    }
}
