package com.example.graduateserver.controller;

import com.example.graduateserver.dto.ProblemRequest;
import com.example.graduateserver.dto.ProblemResponse;
import org.springframework.http.HttpStatus;
import org.springframework.http.ResponseEntity;
import org.springframework.web.bind.annotation.CrossOrigin;
import org.springframework.web.bind.annotation.PostMapping;
import org.springframework.web.bind.annotation.RequestBody;
import org.springframework.web.bind.annotation.RequestMapping;
import org.springframework.web.bind.annotation.RestController;

@RestController
@RequestMapping("/api/test")
@CrossOrigin(origins = "*")
public class TestController {
    @PostMapping
    public ResponseEntity<String> testResponse(@RequestBody ProblemRequest problemRequest) {
        return new ResponseEntity<>("C", HttpStatus.OK);
    }
}
