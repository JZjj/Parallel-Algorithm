package com.example.springbootapp.controller;

import com.example.springbootapp.config.AppConfig;
import com.example.springbootapp.dto.PagedResponseDTO;
import com.example.springbootapp.dto.UserCreateDTO;
import com.example.springbootapp.dto.UserResponseDTO;
import com.example.springbootapp.dto.UserUpdateDTO;
import com.example.springbootapp.service.UserService;
import io.swagger.v3.oas.annotations.Operation;
import io.swagger.v3.oas.annotations.Parameter;
import io.swagger.v3.oas.annotations.media.Content;
import io.swagger.v3.oas.annotations.media.Schema;
import io.swagger.v3.oas.annotations.responses.ApiResponse;
import io.swagger.v3.oas.annotations.responses.ApiResponses;
import io.swagger.v3.oas.annotations.tags.Tag;
import jakarta.validation.Valid;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;
import org.springframework.http.HttpStatus;
import org.springframework.http.ResponseEntity;
import org.springframework.web.bind.annotation.*;

import java.net.URI;
import java.time.LocalDateTime;
import java.util.List;
import java.util.Map;

/**
 * REST Controller for User management operations.
 * 
 * Provides CRUD endpoints for user management with proper HTTP status codes,
 * validation, error handling, and API documentation.
 */
@RestController
@RequestMapping("/users")
@Tag(name = "User Management", description = "APIs for managing users")
public class UserController {

    private static final Logger logger = LoggerFactory.getLogger(UserController.class);

    private final UserService userService;
    private final AppConfig.ApplicationCounter applicationCounter;

    /**
     * Constructor-based dependency injection.
     * 
     * @param userService the user service
     * @param applicationCounter application counter bean
     */
    public UserController(UserService userService, AppConfig.ApplicationCounter applicationCounter) {
        this.userService = userService;
        this.applicationCounter = applicationCounter;
    }

    @Operation(summary = "Create a new user", description = "Creates a new user with the provided information")
    @ApiResponses(value = {
            @ApiResponse(responseCode = "201", description = "User created successfully",
                    content = @Content(schema = @Schema(implementation = UserResponseDTO.class))),
            @ApiResponse(responseCode = "400", description = "Invalid input data"),
            @ApiResponse(responseCode = "409", description = "User with email already exists")
    })
    @PostMapping
    public ResponseEntity<UserResponseDTO> createUser(
            @Valid @RequestBody UserCreateDTO userCreateDTO) {
        
        logger.info("Creating user with email: {}", userCreateDTO.getEmail());
        applicationCounter.increment();

        UserResponseDTO createdUser = userService.createUser(userCreateDTO);
        
        // Create location URI for the created resource
        URI location = URI.create("/users/" + createdUser.getId());
        
        return ResponseEntity.created(location).body(createdUser);
    }

    @Operation(summary = "Get user by ID", description = "Retrieves a user by their unique identifier")
    @ApiResponses(value = {
            @ApiResponse(responseCode = "200", description = "User found",
                    content = @Content(schema = @Schema(implementation = UserResponseDTO.class))),
            @ApiResponse(responseCode = "404", description = "User not found")
    })
    @GetMapping("/{id}")
    public ResponseEntity<UserResponseDTO> getUserById(
            @Parameter(description = "User ID", example = "123e4567-e89b-12d3-a456-426614174000")
            @PathVariable String id) {
        
        logger.debug("Getting user by id: {}", id);
        applicationCounter.increment();

        UserResponseDTO user = userService.getUserById(id);
        return ResponseEntity.ok(user);
    }

    @Operation(summary = "Get all users", description = "Retrieves all users with optional pagination")
    @ApiResponses(value = {
            @ApiResponse(responseCode = "200", description = "Users retrieved successfully")
    })
    @GetMapping
    public ResponseEntity<PagedResponseDTO<UserResponseDTO>> getAllUsers(
            @Parameter(description = "Page number (0-based)", example = "0")
            @RequestParam(defaultValue = "0") int page,
            @Parameter(description = "Number of items per page", example = "10")
            @RequestParam(defaultValue = "10") int size) {
        
        logger.debug("Getting all users - page: {}, size: {}", page, size);
        applicationCounter.increment();

        PagedResponseDTO<UserResponseDTO> users = userService.getAllUsers(page, size);
        return ResponseEntity.ok(users);
    }

    @Operation(summary = "Get all users (no pagination)", description = "Retrieves all users without pagination")
    @ApiResponses(value = {
            @ApiResponse(responseCode = "200", description = "Users retrieved successfully")
    })
    @GetMapping("/all")
    public ResponseEntity<List<UserResponseDTO>> getAllUsersNoPagination() {
        logger.debug("Getting all users without pagination");
        applicationCounter.increment();

        List<UserResponseDTO> users = userService.getAllUsers();
        return ResponseEntity.ok(users);
    }

    @Operation(summary = "Update user", description = "Updates an existing user with the provided information")
    @ApiResponses(value = {
            @ApiResponse(responseCode = "200", description = "User updated successfully",
                    content = @Content(schema = @Schema(implementation = UserResponseDTO.class))),
            @ApiResponse(responseCode = "400", description = "Invalid input data"),
            @ApiResponse(responseCode = "404", description = "User not found"),
            @ApiResponse(responseCode = "409", description = "Email already exists for another user")
    })
    @PutMapping("/{id}")
    public ResponseEntity<UserResponseDTO> updateUser(
            @Parameter(description = "User ID", example = "123e4567-e89b-12d3-a456-426614174000")
            @PathVariable String id,
            @Valid @RequestBody UserUpdateDTO userUpdateDTO) {
        
        logger.info("Updating user with id: {}", id);
        applicationCounter.increment();

        UserResponseDTO updatedUser = userService.updateUser(id, userUpdateDTO);
        return ResponseEntity.ok(updatedUser);
    }

    @Operation(summary = "Delete user", description = "Deletes a user by their unique identifier")
    @ApiResponses(value = {
            @ApiResponse(responseCode = "204", description = "User deleted successfully"),
            @ApiResponse(responseCode = "404", description = "User not found")
    })
    @DeleteMapping("/{id}")
    public ResponseEntity<Void> deleteUser(
            @Parameter(description = "User ID", example = "123e4567-e89b-12d3-a456-426614174000")
            @PathVariable String id) {
        
        logger.info("Deleting user with id: {}", id);
        applicationCounter.increment();

        userService.deleteUser(id);
        return ResponseEntity.noContent().build();
    }

    @Operation(summary = "Check if user exists", description = "Checks if a user exists by their unique identifier")
    @ApiResponses(value = {
            @ApiResponse(responseCode = "200", description = "Check completed"),
            @ApiResponse(responseCode = "404", description = "User not found")
    })
    @GetMapping("/{id}/exists")
    public ResponseEntity<Map<String, Boolean>> checkUserExists(
            @Parameter(description = "User ID", example = "123e4567-e89b-12d3-a456-426614174000")
            @PathVariable String id) {
        
        logger.debug("Checking if user exists with id: {}", id);
        applicationCounter.increment();

        boolean exists = userService.existsById(id);
        return ResponseEntity.ok(Map.of("exists", exists));
    }

    @Operation(summary = "Get application statistics", description = "Returns application usage statistics")
    @ApiResponses(value = {
            @ApiResponse(responseCode = "200", description = "Statistics retrieved successfully")
    })
    @GetMapping("/stats")
    public ResponseEntity<Map<String, Object>> getApplicationStats() {
        logger.debug("Getting application statistics");

        Map<String, Object> stats = Map.of(
                "totalUsers", userService.getUserCount(),
                "requestCount", applicationCounter.getCount(),
                "applicationStartTime", applicationCounter.getCreatedAt(),
                "currentTime", LocalDateTime.now()
        );

        return ResponseEntity.ok(stats);
    }
}