// generated from rosidl_generator_c/resource/idl__struct.h.em
// with input from strawberry_msgs:srv/CaptureSnapshot.idl
// generated code does not contain a copyright notice

#ifndef STRAWBERRY_MSGS__SRV__DETAIL__CAPTURE_SNAPSHOT__STRUCT_H_
#define STRAWBERRY_MSGS__SRV__DETAIL__CAPTURE_SNAPSHOT__STRUCT_H_

#ifdef __cplusplus
extern "C"
{
#endif

#include <stdbool.h>
#include <stddef.h>
#include <stdint.h>


// Constants defined in the message

/// Struct defined in srv/CaptureSnapshot in the package strawberry_msgs.
typedef struct strawberry_msgs__srv__CaptureSnapshot_Request
{
  int32_t plant_id;
  int32_t view_id;
} strawberry_msgs__srv__CaptureSnapshot_Request;

// Struct for a sequence of strawberry_msgs__srv__CaptureSnapshot_Request.
typedef struct strawberry_msgs__srv__CaptureSnapshot_Request__Sequence
{
  strawberry_msgs__srv__CaptureSnapshot_Request * data;
  /// The number of valid items in data
  size_t size;
  /// The number of allocated items in data
  size_t capacity;
} strawberry_msgs__srv__CaptureSnapshot_Request__Sequence;


// Constants defined in the message

// Include directives for member types
// Member 'message'
// Member 'rgb_path'
// Member 'depth_path'
#include "rosidl_runtime_c/string.h"

/// Struct defined in srv/CaptureSnapshot in the package strawberry_msgs.
typedef struct strawberry_msgs__srv__CaptureSnapshot_Response
{
  bool success;
  rosidl_runtime_c__String message;
  rosidl_runtime_c__String rgb_path;
  rosidl_runtime_c__String depth_path;
} strawberry_msgs__srv__CaptureSnapshot_Response;

// Struct for a sequence of strawberry_msgs__srv__CaptureSnapshot_Response.
typedef struct strawberry_msgs__srv__CaptureSnapshot_Response__Sequence
{
  strawberry_msgs__srv__CaptureSnapshot_Response * data;
  /// The number of valid items in data
  size_t size;
  /// The number of allocated items in data
  size_t capacity;
} strawberry_msgs__srv__CaptureSnapshot_Response__Sequence;

#ifdef __cplusplus
}
#endif

#endif  // STRAWBERRY_MSGS__SRV__DETAIL__CAPTURE_SNAPSHOT__STRUCT_H_
