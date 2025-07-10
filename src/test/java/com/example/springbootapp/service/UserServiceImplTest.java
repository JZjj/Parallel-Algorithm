package com.example.springbootapp.service;

import com.example.springbootapp.dto.UserCreateDTO;
import com.example.springbootapp.dto.UserResponseDTO;
import com.example.springbootapp.dto.UserUpdateDTO;
import com.example.springbootapp.exception.UserAlreadyExistsException;
import com.example.springbootapp.exception.UserNotFoundException;
import com.example.springbootapp.model.User;
import com.example.springbootapp.repository.UserRepository;
import org.junit.jupiter.api.BeforeEach;
import org.junit.jupiter.api.Test;
import org.junit.jupiter.api.extension.ExtendWith;
import org.mockito.InjectMocks;
import org.mockito.Mock;
import org.mockito.junit.jupiter.MockitoExtension;

import java.time.LocalDateTime;
import java.util.Optional;

import static org.assertj.core.api.Assertions.*;
import static org.mockito.ArgumentMatchers.any;
import static org.mockito.ArgumentMatchers.anyString;
import static org.mockito.Mockito.*;

/**
 * Unit tests for UserServiceImpl.
 */
@ExtendWith(MockitoExtension.class)
class UserServiceImplTest {

    @Mock
    private UserRepository userRepository;

    @InjectMocks
    private UserServiceImpl userService;

    private UserCreateDTO userCreateDTO;
    private User user;

    @BeforeEach
    void setUp() {
        userCreateDTO = new UserCreateDTO("test@example.com", "John", "Doe", 25);
        
        user = new User();
        user.setId("test-id");
        user.setEmail("test@example.com");
        user.setFirstName("John");
        user.setLastName("Doe");
        user.setAge(25);
        user.setCreatedAt(LocalDateTime.now());
        user.setUpdatedAt(LocalDateTime.now());
    }

    @Test
    void createUser_Success() {
        // Given
        when(userRepository.existsByEmail(anyString())).thenReturn(false);
        when(userRepository.save(any(User.class))).thenReturn(user);

        // When
        UserResponseDTO result = userService.createUser(userCreateDTO);

        // Then
        assertThat(result).isNotNull();
        assertThat(result.getEmail()).isEqualTo("test@example.com");
        assertThat(result.getFirstName()).isEqualTo("John");
        assertThat(result.getLastName()).isEqualTo("Doe");
        assertThat(result.getAge()).isEqualTo(25);

        verify(userRepository).existsByEmail("test@example.com");
        verify(userRepository).save(any(User.class));
    }

    @Test
    void createUser_EmailAlreadyExists_ThrowsException() {
        // Given
        when(userRepository.existsByEmail(anyString())).thenReturn(true);

        // When & Then
        assertThatThrownBy(() -> userService.createUser(userCreateDTO))
                .isInstanceOf(UserAlreadyExistsException.class)
                .hasMessageContaining("test@example.com");

        verify(userRepository).existsByEmail("test@example.com");
        verify(userRepository, never()).save(any(User.class));
    }

    @Test
    void getUserById_Success() {
        // Given
        when(userRepository.findById("test-id")).thenReturn(Optional.of(user));

        // When
        UserResponseDTO result = userService.getUserById("test-id");

        // Then
        assertThat(result).isNotNull();
        assertThat(result.getId()).isEqualTo("test-id");
        assertThat(result.getEmail()).isEqualTo("test@example.com");

        verify(userRepository).findById("test-id");
    }

    @Test
    void getUserById_NotFound_ThrowsException() {
        // Given
        when(userRepository.findById("non-existent")).thenReturn(Optional.empty());

        // When & Then
        assertThatThrownBy(() -> userService.getUserById("non-existent"))
                .isInstanceOf(UserNotFoundException.class)
                .hasMessageContaining("non-existent");

        verify(userRepository).findById("non-existent");
    }

    @Test
    void updateUser_Success() {
        // Given
        UserUpdateDTO updateDTO = new UserUpdateDTO("newemail@example.com", "Jane", null, 30);
        when(userRepository.findById("test-id")).thenReturn(Optional.of(user));
        when(userRepository.findByEmail("newemail@example.com")).thenReturn(Optional.empty());
        when(userRepository.save(any(User.class))).thenReturn(user);

        // When
        UserResponseDTO result = userService.updateUser("test-id", updateDTO);

        // Then
        assertThat(result).isNotNull();
        verify(userRepository).findById("test-id");
        verify(userRepository).findByEmail("newemail@example.com");
        verify(userRepository).save(any(User.class));
    }

    @Test
    void updateUser_NoUpdates_ThrowsException() {
        // Given
        UserUpdateDTO emptyUpdateDTO = new UserUpdateDTO();

        // When & Then
        assertThatThrownBy(() -> userService.updateUser("test-id", emptyUpdateDTO))
                .isInstanceOf(IllegalArgumentException.class)
                .hasMessageContaining("No fields provided for update");

        verify(userRepository, never()).findById(anyString());
    }

    @Test
    void deleteUser_Success() {
        // Given
        when(userRepository.existsById("test-id")).thenReturn(true);

        // When
        userService.deleteUser("test-id");

        // Then
        verify(userRepository).existsById("test-id");
        verify(userRepository).deleteById("test-id");
    }

    @Test
    void deleteUser_NotFound_ThrowsException() {
        // Given
        when(userRepository.existsById("non-existent")).thenReturn(false);

        // When & Then
        assertThatThrownBy(() -> userService.deleteUser("non-existent"))
                .isInstanceOf(UserNotFoundException.class)
                .hasMessageContaining("non-existent");

        verify(userRepository).existsById("non-existent");
        verify(userRepository, never()).deleteById(anyString());
    }
}