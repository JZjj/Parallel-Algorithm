package com.example.springbootapp.controller;

import com.example.springbootapp.config.AppConfig;
import com.example.springbootapp.dto.UserCreateDTO;
import com.example.springbootapp.dto.UserResponseDTO;
import com.example.springbootapp.dto.UserUpdateDTO;
import com.example.springbootapp.service.UserService;
import com.fasterxml.jackson.databind.ObjectMapper;
import org.junit.jupiter.api.Test;
import org.springframework.beans.factory.annotation.Autowired;
import org.springframework.boot.test.autoconfigure.web.servlet.WebMvcTest;
import org.springframework.boot.test.mock.mockito.MockBean;
import org.springframework.http.MediaType;
import org.springframework.test.web.servlet.MockMvc;

import java.time.LocalDateTime;

import static org.mockito.ArgumentMatchers.*;
import static org.mockito.Mockito.when;
import static org.springframework.test.web.servlet.request.MockMvcRequestBuilders.*;
import static org.springframework.test.web.servlet.result.MockMvcResultMatchers.*;

/**
 * Unit tests for UserController.
 */
@WebMvcTest(UserController.class)
class UserControllerTest {

    @Autowired
    private MockMvc mockMvc;

    @Autowired
    private ObjectMapper objectMapper;

    @MockBean
    private UserService userService;

    @MockBean
    private AppConfig.ApplicationCounter applicationCounter;

    @Test
    void createUser_Success() throws Exception {
        // Given
        UserCreateDTO createDTO = new UserCreateDTO("test@example.com", "John", "Doe", 25);
        UserResponseDTO responseDTO = new UserResponseDTO(
                "test-id", "test@example.com", "John", "Doe", 25,
                LocalDateTime.now(), LocalDateTime.now()
        );

        when(userService.createUser(any(UserCreateDTO.class))).thenReturn(responseDTO);

        // When & Then
        mockMvc.perform(post("/users")
                        .contentType(MediaType.APPLICATION_JSON)
                        .content(objectMapper.writeValueAsString(createDTO)))
                .andExpect(status().isCreated())
                .andExpect(header().string("Location", "/users/test-id"))
                .andExpect(jsonPath("$.id").value("test-id"))
                .andExpect(jsonPath("$.email").value("test@example.com"))
                .andExpect(jsonPath("$.firstName").value("John"))
                .andExpect(jsonPath("$.lastName").value("Doe"))
                .andExpect(jsonPath("$.age").value(25));
    }

    @Test
    void createUser_InvalidInput_ReturnsBadRequest() throws Exception {
        // Given - invalid user data (missing required fields)
        UserCreateDTO invalidCreateDTO = new UserCreateDTO("", "", "", null);

        // When & Then
        mockMvc.perform(post("/users")
                        .contentType(MediaType.APPLICATION_JSON)
                        .content(objectMapper.writeValueAsString(invalidCreateDTO)))
                .andExpect(status().isBadRequest());
    }

    @Test
    void getUserById_Success() throws Exception {
        // Given
        UserResponseDTO responseDTO = new UserResponseDTO(
                "test-id", "test@example.com", "John", "Doe", 25,
                LocalDateTime.now(), LocalDateTime.now()
        );

        when(userService.getUserById("test-id")).thenReturn(responseDTO);

        // When & Then
        mockMvc.perform(get("/users/test-id"))
                .andExpect(status().isOk())
                .andExpect(jsonPath("$.id").value("test-id"))
                .andExpect(jsonPath("$.email").value("test@example.com"));
    }

    @Test
    void updateUser_Success() throws Exception {
        // Given
        UserUpdateDTO updateDTO = new UserUpdateDTO("newemail@example.com", "Jane", null, 30);
        UserResponseDTO responseDTO = new UserResponseDTO(
                "test-id", "newemail@example.com", "Jane", "Doe", 30,
                LocalDateTime.now(), LocalDateTime.now()
        );

        when(userService.updateUser(eq("test-id"), any(UserUpdateDTO.class))).thenReturn(responseDTO);

        // When & Then
        mockMvc.perform(put("/users/test-id")
                        .contentType(MediaType.APPLICATION_JSON)
                        .content(objectMapper.writeValueAsString(updateDTO)))
                .andExpect(status().isOk())
                .andExpect(jsonPath("$.email").value("newemail@example.com"))
                .andExpect(jsonPath("$.firstName").value("Jane"))
                .andExpect(jsonPath("$.age").value(30));
    }

    @Test
    void deleteUser_Success() throws Exception {
        // When & Then
        mockMvc.perform(delete("/users/test-id"))
                .andExpect(status().isNoContent());
    }

    @Test
    void checkUserExists_Success() throws Exception {
        // Given
        when(userService.existsById("test-id")).thenReturn(true);

        // When & Then
        mockMvc.perform(get("/users/test-id/exists"))
                .andExpect(status().isOk())
                .andExpect(jsonPath("$.exists").value(true));
    }

    @Test
    void getApplicationStats_Success() throws Exception {
        // Given
        when(userService.getUserCount()).thenReturn(5L);
        when(applicationCounter.getCount()).thenReturn(100L);
        when(applicationCounter.getCreatedAt()).thenReturn(LocalDateTime.now());

        // When & Then
        mockMvc.perform(get("/users/stats"))
                .andExpect(status().isOk())
                .andExpect(jsonPath("$.totalUsers").value(5))
                .andExpect(jsonPath("$.requestCount").value(100));
    }
}