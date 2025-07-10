package com.example.springbootapp;

import io.swagger.v3.oas.annotations.OpenAPIDefinition;
import io.swagger.v3.oas.annotations.info.Contact;
import io.swagger.v3.oas.annotations.info.Info;
import io.swagger.v3.oas.annotations.info.License;
import org.springframework.boot.SpringApplication;
import org.springframework.boot.autoconfigure.SpringBootApplication;

/**
 * Main Spring Boot Application class.
 * 
 * This application demonstrates best practices with:
 * - REST API endpoints
 * - DTO definitions with validation
 * - Service layer with dependency injection
 * - Custom bean configurations
 * - Redis integration
 * - Comprehensive error handling
 * - Swagger/OpenAPI documentation
 */
@SpringBootApplication
@OpenAPIDefinition(
    info = @Info(
        title = "Spring Boot User Management API",
        version = "1.0.0",
        description = "A comprehensive Spring Boot application demonstrating best practices with REST APIs, Redis integration, and dependency injection",
        contact = @Contact(
            name = "Spring Boot App Team",
            email = "support@example.com"
        ),
        license = @License(
            name = "MIT License",
            url = "https://opensource.org/licenses/MIT"
        )
    )
)
public class SpringBootAppApplication {

    public static void main(String[] args) {
        SpringApplication.run(SpringBootAppApplication.class, args);
    }
}