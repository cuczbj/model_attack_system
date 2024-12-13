package com.example.demo;

import org.springframework.web.bind.annotation.GetMapping;
import org.springframework.web.bind.annotation.RequestMapping;
import org.springframework.web.bind.annotation.RestController;

@RestController
@RequestMapping("/api") // 定义 API 基础路径
public class HelloController {

    @GetMapping("/hello") // 定义 GET 请求路径为
    public String sayHello() {
        return "Hello, Spring Boot!";
    }

}
