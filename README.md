# Spring Boot User Management API

A comprehensive Spring Boot application demonstrating best practices with REST API endpoints, Redis integration, dependency injection, custom bean configurations, and extensive testing.

## Features

- **REST API**: Complete CRUD operations for user management
- **Redis Integration**: Using Spring Data Redis for data persistence and caching
- **DTO Validation**: Comprehensive input validation with Jakarta Bean Validation
- **Dependency Injection**: Constructor-based dependency injection following best practices
- **Custom Bean Configurations**: Singleton and prototype scoped beans with lifecycle methods
- **Exception Handling**: Global exception handling with proper HTTP status codes
- **API Documentation**: Swagger/OpenAPI integration for interactive API documentation
- **Testing**: Unit tests, integration tests with Testcontainers
- **Logging**: Structured logging with different levels for development and production

## Technology Stack

- **Java 17**
- **Spring Boot 3.2.0**
- **Spring Data Redis**
- **Spring Boot Validation**
- **Maven**
- **Redis 7.0+**
- **Swagger/OpenAPI 3**
- **JUnit 5**
- **Testcontainers**
- **AssertJ**
- **Mockito**

## Prerequisites

- Java 17 or higher
- Maven 3.6+
- Redis 7.0+ (or use Docker)

## Quick Start

### 1. Clone the Repository

```bash
git clone <repository-url>
cd Parallel-Algorithm
```

### 2. Start Redis

#### Using Docker
```bash
docker run -d --name redis -p 6379:6379 redis:7.0.12-alpine
```

#### Using Local Installation
Make sure Redis is running on `localhost:6379`

### 3. Build and Run

```bash
# Build the application
mvn clean compile

# Run the application
mvn spring-boot:run
```

The application will start on `http://localhost:8080`

### 4. Access API Documentation

Open your browser and navigate to:
- **Swagger UI**: http://localhost:8080/api/v1/swagger-ui.html
- **OpenAPI JSON**: http://localhost:8080/api/v1/api-docs

## API Endpoints

| Method | Endpoint | Description |
|--------|----------|-------------|
| POST | `/api/v1/users` | Create a new user |
| GET | `/api/v1/users/{id}` | Get user by ID |
| GET | `/api/v1/users` | Get all users (paginated) |
| GET | `/api/v1/users/all` | Get all users (no pagination) |
| PUT | `/api/v1/users/{id}` | Update user |
| DELETE | `/api/v1/users/{id}` | Delete user |
| GET | `/api/v1/users/{id}/exists` | Check if user exists |
| GET | `/api/v1/users/stats` | Get application statistics |

### Example API Usage

#### Create a User
```bash
curl -X POST http://localhost:8080/api/v1/users \
  -H "Content-Type: application/json" \
  -d '{
    "email": "john.doe@example.com",
    "firstName": "John",
    "lastName": "Doe",
    "age": 30
  }'
```

#### Get All Users (Paginated)
```bash
curl "http://localhost:8080/api/v1/users?page=0&size=10"
```

#### Update a User
```bash
curl -X PUT http://localhost:8080/api/v1/users/{user-id} \
  -H "Content-Type: application/json" \
  -d '{
    "firstName": "Jane",
    "age": 25
  }'
```

## Configuration

### Application Configuration

The application can be configured via `application.yml`:

```yaml
server:
  port: 8080
  servlet:
    context-path: /api/v1

spring:
  data:
    redis:
      host: localhost
      port: 6379
      database: 0
```

### Profiles

- **Default**: Uses local Redis instance
- **Test**: Uses test-specific configurations

## Data Model

### User Entity

```json
{
  "id": "string (UUID)",
  "email": "string (unique, validated)",
  "firstName": "string (2-50 chars)",
  "lastName": "string (2-50 chars)", 
  "age": "integer (18-120)",
  "createdAt": "datetime",
  "updatedAt": "datetime"
}
```

### Validation Rules

- **Email**: Required, must be valid email format, max 100 characters
- **First Name**: Required, 2-50 characters
- **Last Name**: Required, 2-50 characters  
- **Age**: Required, must be between 18-120

## Testing

### Run All Tests
```bash
mvn test
```

### Run Specific Test Categories

```bash
# Unit tests only
mvn test -Dtest="*Test"

# Integration tests only  
mvn test -Dtest="*IntegrationTest"
```

### Test Coverage

The application includes:
- **Unit Tests**: Service and controller layer testing with mocking
- **Integration Tests**: End-to-end testing with Testcontainers for Redis
- **Validation Tests**: Input validation and error handling

## Architecture

### Package Structure

```
src/main/java/com/example/springbootapp/
├── SpringBootAppApplication.java     # Main application class
├── controller/                       # REST controllers
│   └── UserController.java
├── dto/                             # Data Transfer Objects
│   ├── UserCreateDTO.java
│   ├── UserUpdateDTO.java
│   ├── UserResponseDTO.java
│   └── PagedResponseDTO.java
├── service/                         # Business logic layer
│   ├── UserService.java
│   └── UserServiceImpl.java
├── repository/                      # Data access layer
│   └── UserRepository.java
├── model/                          # Entity models
│   └── User.java
├── config/                         # Configuration classes
│   ├── RedisConfig.java
│   └── AppConfig.java
└── exception/                      # Exception handling
    ├── GlobalExceptionHandler.java
    ├── UserNotFoundException.java
    ├── UserAlreadyExistsException.java
    └── ErrorResponseDTO.java
```

### Design Patterns Used

1. **Repository Pattern**: Data access abstraction
2. **Service Layer Pattern**: Business logic separation
3. **DTO Pattern**: Data transfer between layers
4. **Dependency Injection**: Constructor-based injection
5. **Factory Pattern**: Custom bean configurations
6. **Exception Handling**: Global exception handling

## Monitoring and Observability

### Health Checks

The application exposes actuator endpoints:
- **Health**: `/api/v1/actuator/health`
- **Info**: `/api/v1/actuator/info`
- **Metrics**: `/api/v1/actuator/metrics`

### Application Statistics

Get real-time application statistics via:
```bash
curl http://localhost:8080/api/v1/users/stats
```

## Custom Beans and Dependency Injection

The application demonstrates various Spring concepts:

- **Singleton Beans**: Application-wide counters
- **Prototype Beans**: Per-request instances
- **Bean Lifecycle**: Initialization and destruction methods
- **Constructor Injection**: Preferred dependency injection method

## Redis Integration

### Features Implemented

- **Entity Mapping**: Using `@RedisHash` for entity storage
- **Indexing**: Efficient querying with `@Indexed` fields
- **Serialization**: JSON serialization with Jackson
- **Connection Pooling**: Lettuce connection factory with pooling
- **Custom Templates**: Redis template configurations

### Data Storage

Users are stored in Redis with:
- **Key Pattern**: `users:{id}`
- **Indexed Fields**: email for efficient lookups
- **TTL**: No expiration (persistent storage)

## Error Handling

### HTTP Status Codes

- **200 OK**: Successful GET/PUT operations
- **201 Created**: Successful POST operations
- **204 No Content**: Successful DELETE operations
- **400 Bad Request**: Validation errors
- **404 Not Found**: Resource not found
- **409 Conflict**: Resource conflicts (duplicate email)
- **500 Internal Server Error**: Unexpected errors

### Error Response Format

```json
{
  "timestamp": "2024-01-15T10:30:00",
  "status": 404,
  "error": "NOT_FOUND",
  "message": "User not found with id: 123",
  "path": "/api/v1/users/123",
  "validationErrors": ["field: error message"]
}
```

## Development

### Code Style

- Follow Spring Boot best practices
- Constructor-based dependency injection
- Comprehensive validation
- Proper error handling
- Extensive logging

### Adding New Features

1. Create DTOs in `dto` package
2. Add validation annotations
3. Implement service methods
4. Add repository methods if needed
5. Create controller endpoints
6. Add comprehensive tests
7. Update API documentation

## Troubleshooting

### Common Issues

1. **Redis Connection Error**
   - Ensure Redis is running on localhost:6379
   - Check Redis configuration in application.yml

2. **Port Already in Use**
   - Change server port in application.yml
   - Kill existing processes using port 8080

3. **Test Failures**
   - Ensure Docker is running for Testcontainers
   - Check Redis availability for integration tests

### Logging

Enable debug logging by setting:
```yaml
logging:
  level:
    com.example.springbootapp: DEBUG
```

## Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests for new functionality
5. Ensure all tests pass
6. Create a pull request

## License

This project is licensed under the MIT License - see the LICENSE file for details.
