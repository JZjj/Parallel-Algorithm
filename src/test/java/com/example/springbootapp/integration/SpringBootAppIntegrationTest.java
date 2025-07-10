package com.example.springbootapp.integration;

import com.example.springbootapp.SpringBootAppApplication;
import com.example.springbootapp.dto.UserCreateDTO;
import com.example.springbootapp.dto.UserResponseDTO;
import com.fasterxml.jackson.databind.ObjectMapper;
import com.redis.testcontainers.RedisContainer;
import org.junit.jupiter.api.BeforeEach;
import org.junit.jupiter.api.Test;
import org.springframework.beans.factory.annotation.Autowired;

import org.springframework.boot.test.context.SpringBootTest;
import org.springframework.boot.test.web.client.TestRestTemplate;
import org.springframework.boot.test.web.server.LocalServerPort;
import org.springframework.http.HttpStatus;
import org.springframework.http.ResponseEntity;
import org.springframework.test.context.DynamicPropertyRegistry;
import org.springframework.test.context.DynamicPropertySource;
import org.springframework.test.context.TestPropertySource;
import org.testcontainers.junit.jupiter.Container;
import org.testcontainers.junit.jupiter.Testcontainers;
import org.testcontainers.utility.DockerImageName;

import static org.assertj.core.api.Assertions.assertThat;

/**
 * Integration tests for the Spring Boot application using Testcontainers.
 */
@SpringBootTest(
        classes = SpringBootAppApplication.class,
        webEnvironment = SpringBootTest.WebEnvironment.RANDOM_PORT
)
@Testcontainers
@TestPropertySource(locations = "classpath:application-test.yml")
class SpringBootAppIntegrationTest {

    @Container
    static RedisContainer redisContainer = new RedisContainer(DockerImageName.parse("redis:7.0.12-alpine"))
            .withExposedPorts(6379);

    @DynamicPropertySource
    static void configureProperties(DynamicPropertyRegistry registry) {
        registry.add("spring.data.redis.host", redisContainer::getHost);
        registry.add("spring.data.redis.port", redisContainer::getFirstMappedPort);
    }

    @LocalServerPort
    private int port;

    @Autowired
    private TestRestTemplate restTemplate;

    private String baseUrl;

    @BeforeEach
    void setUp() {
        baseUrl = "http://localhost:" + port + "/api/v1/users";
    }

    @Test
    void contextLoads() {
        assertThat(redisContainer.isRunning()).isTrue();
    }

    @Test
    void createAndRetrieveUser_Success() {
        // Given
        UserCreateDTO createDTO = new UserCreateDTO(
                "integration@example.com", "Integration", "Test", 30
        );

        // When - Create user
        ResponseEntity<UserResponseDTO> createResponse = restTemplate.postForEntity(
                baseUrl, createDTO, UserResponseDTO.class
        );

        // Then - Verify creation
        assertThat(createResponse.getStatusCode()).isEqualTo(HttpStatus.CREATED);
        assertThat(createResponse.getBody()).isNotNull();
        assertThat(createResponse.getBody().getEmail()).isEqualTo("integration@example.com");

        String userId = createResponse.getBody().getId();

        // When - Retrieve user
        ResponseEntity<UserResponseDTO> getResponse = restTemplate.getForEntity(
                baseUrl + "/" + userId, UserResponseDTO.class
        );

        // Then - Verify retrieval
        assertThat(getResponse.getStatusCode()).isEqualTo(HttpStatus.OK);
        assertThat(getResponse.getBody()).isNotNull();
        assertThat(getResponse.getBody().getId()).isEqualTo(userId);
        assertThat(getResponse.getBody().getEmail()).isEqualTo("integration@example.com");
    }

    @Test
    void createUser_DuplicateEmail_ReturnsConflict() {
        // Given
        UserCreateDTO createDTO = new UserCreateDTO(
                "duplicate@example.com", "Duplicate", "User", 25
        );

        // When - Create first user
        ResponseEntity<UserResponseDTO> firstResponse = restTemplate.postForEntity(
                baseUrl, createDTO, UserResponseDTO.class
        );

        // Then - First creation should succeed
        assertThat(firstResponse.getStatusCode()).isEqualTo(HttpStatus.CREATED);

        // When - Try to create second user with same email
        ResponseEntity<String> secondResponse = restTemplate.postForEntity(
                baseUrl, createDTO, String.class
        );

        // Then - Second creation should fail
        assertThat(secondResponse.getStatusCode()).isEqualTo(HttpStatus.CONFLICT);
    }

    @Test
    void getUserById_NotFound_ReturnsNotFound() {
        // When
        ResponseEntity<String> response = restTemplate.getForEntity(
                baseUrl + "/non-existent-id", String.class
        );

        // Then
        assertThat(response.getStatusCode()).isEqualTo(HttpStatus.NOT_FOUND);
    }

    @Test
    void deleteUser_Success() {
        // Given - Create a user first
        UserCreateDTO createDTO = new UserCreateDTO(
                "delete@example.com", "Delete", "Me", 25
        );
        ResponseEntity<UserResponseDTO> createResponse = restTemplate.postForEntity(
                baseUrl, createDTO, UserResponseDTO.class
        );
        
        assertThat(createResponse.getStatusCode()).isEqualTo(HttpStatus.CREATED);
        String userId = createResponse.getBody().getId();

        // When - Delete the user
        restTemplate.delete(baseUrl + "/" + userId);

        // Then - Verify user is deleted
        ResponseEntity<String> getResponse = restTemplate.getForEntity(
                baseUrl + "/" + userId, String.class
        );
        assertThat(getResponse.getStatusCode()).isEqualTo(HttpStatus.NOT_FOUND);
    }
}