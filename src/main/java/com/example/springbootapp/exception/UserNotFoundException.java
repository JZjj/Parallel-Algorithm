package com.example.springbootapp.exception;

/**
 * Custom exception thrown when a user is not found in the system.
 */
public class UserNotFoundException extends RuntimeException {

    public UserNotFoundException(String id) {
        super("User not found with id: " + id);
    }

    public UserNotFoundException(String message, Throwable cause) {
        super(message, cause);
    }
}