// generated from rosidl_typesupport_introspection_c/resource/idl__type_support.c.em
// with input from strawberry_msgs:srv/CaptureSnapshot.idl
// generated code does not contain a copyright notice

#include <stddef.h>
#include "strawberry_msgs/srv/detail/capture_snapshot__rosidl_typesupport_introspection_c.h"
#include "strawberry_msgs/msg/rosidl_typesupport_introspection_c__visibility_control.h"
#include "rosidl_typesupport_introspection_c/field_types.h"
#include "rosidl_typesupport_introspection_c/identifier.h"
#include "rosidl_typesupport_introspection_c/message_introspection.h"
#include "strawberry_msgs/srv/detail/capture_snapshot__functions.h"
#include "strawberry_msgs/srv/detail/capture_snapshot__struct.h"


#ifdef __cplusplus
extern "C"
{
#endif

void strawberry_msgs__srv__CaptureSnapshot_Request__rosidl_typesupport_introspection_c__CaptureSnapshot_Request_init_function(
  void * message_memory, enum rosidl_runtime_c__message_initialization _init)
{
  // TODO(karsten1987): initializers are not yet implemented for typesupport c
  // see https://github.com/ros2/ros2/issues/397
  (void) _init;
  strawberry_msgs__srv__CaptureSnapshot_Request__init(message_memory);
}

void strawberry_msgs__srv__CaptureSnapshot_Request__rosidl_typesupport_introspection_c__CaptureSnapshot_Request_fini_function(void * message_memory)
{
  strawberry_msgs__srv__CaptureSnapshot_Request__fini(message_memory);
}

static rosidl_typesupport_introspection_c__MessageMember strawberry_msgs__srv__CaptureSnapshot_Request__rosidl_typesupport_introspection_c__CaptureSnapshot_Request_message_member_array[2] = {
  {
    "plant_id",  // name
    rosidl_typesupport_introspection_c__ROS_TYPE_INT32,  // type
    0,  // upper bound of string
    NULL,  // members of sub message
    false,  // is array
    0,  // array size
    false,  // is upper bound
    offsetof(strawberry_msgs__srv__CaptureSnapshot_Request, plant_id),  // bytes offset in struct
    NULL,  // default value
    NULL,  // size() function pointer
    NULL,  // get_const(index) function pointer
    NULL,  // get(index) function pointer
    NULL,  // fetch(index, &value) function pointer
    NULL,  // assign(index, value) function pointer
    NULL  // resize(index) function pointer
  },
  {
    "view_id",  // name
    rosidl_typesupport_introspection_c__ROS_TYPE_INT32,  // type
    0,  // upper bound of string
    NULL,  // members of sub message
    false,  // is array
    0,  // array size
    false,  // is upper bound
    offsetof(strawberry_msgs__srv__CaptureSnapshot_Request, view_id),  // bytes offset in struct
    NULL,  // default value
    NULL,  // size() function pointer
    NULL,  // get_const(index) function pointer
    NULL,  // get(index) function pointer
    NULL,  // fetch(index, &value) function pointer
    NULL,  // assign(index, value) function pointer
    NULL  // resize(index) function pointer
  }
};

static const rosidl_typesupport_introspection_c__MessageMembers strawberry_msgs__srv__CaptureSnapshot_Request__rosidl_typesupport_introspection_c__CaptureSnapshot_Request_message_members = {
  "strawberry_msgs__srv",  // message namespace
  "CaptureSnapshot_Request",  // message name
  2,  // number of fields
  sizeof(strawberry_msgs__srv__CaptureSnapshot_Request),
  strawberry_msgs__srv__CaptureSnapshot_Request__rosidl_typesupport_introspection_c__CaptureSnapshot_Request_message_member_array,  // message members
  strawberry_msgs__srv__CaptureSnapshot_Request__rosidl_typesupport_introspection_c__CaptureSnapshot_Request_init_function,  // function to initialize message memory (memory has to be allocated)
  strawberry_msgs__srv__CaptureSnapshot_Request__rosidl_typesupport_introspection_c__CaptureSnapshot_Request_fini_function  // function to terminate message instance (will not free memory)
};

// this is not const since it must be initialized on first access
// since C does not allow non-integral compile-time constants
static rosidl_message_type_support_t strawberry_msgs__srv__CaptureSnapshot_Request__rosidl_typesupport_introspection_c__CaptureSnapshot_Request_message_type_support_handle = {
  0,
  &strawberry_msgs__srv__CaptureSnapshot_Request__rosidl_typesupport_introspection_c__CaptureSnapshot_Request_message_members,
  get_message_typesupport_handle_function,
};

ROSIDL_TYPESUPPORT_INTROSPECTION_C_EXPORT_strawberry_msgs
const rosidl_message_type_support_t *
ROSIDL_TYPESUPPORT_INTERFACE__MESSAGE_SYMBOL_NAME(rosidl_typesupport_introspection_c, strawberry_msgs, srv, CaptureSnapshot_Request)() {
  if (!strawberry_msgs__srv__CaptureSnapshot_Request__rosidl_typesupport_introspection_c__CaptureSnapshot_Request_message_type_support_handle.typesupport_identifier) {
    strawberry_msgs__srv__CaptureSnapshot_Request__rosidl_typesupport_introspection_c__CaptureSnapshot_Request_message_type_support_handle.typesupport_identifier =
      rosidl_typesupport_introspection_c__identifier;
  }
  return &strawberry_msgs__srv__CaptureSnapshot_Request__rosidl_typesupport_introspection_c__CaptureSnapshot_Request_message_type_support_handle;
}
#ifdef __cplusplus
}
#endif

// already included above
// #include <stddef.h>
// already included above
// #include "strawberry_msgs/srv/detail/capture_snapshot__rosidl_typesupport_introspection_c.h"
// already included above
// #include "strawberry_msgs/msg/rosidl_typesupport_introspection_c__visibility_control.h"
// already included above
// #include "rosidl_typesupport_introspection_c/field_types.h"
// already included above
// #include "rosidl_typesupport_introspection_c/identifier.h"
// already included above
// #include "rosidl_typesupport_introspection_c/message_introspection.h"
// already included above
// #include "strawberry_msgs/srv/detail/capture_snapshot__functions.h"
// already included above
// #include "strawberry_msgs/srv/detail/capture_snapshot__struct.h"


// Include directives for member types
// Member `message`
// Member `rgb_path`
// Member `depth_path`
#include "rosidl_runtime_c/string_functions.h"

#ifdef __cplusplus
extern "C"
{
#endif

void strawberry_msgs__srv__CaptureSnapshot_Response__rosidl_typesupport_introspection_c__CaptureSnapshot_Response_init_function(
  void * message_memory, enum rosidl_runtime_c__message_initialization _init)
{
  // TODO(karsten1987): initializers are not yet implemented for typesupport c
  // see https://github.com/ros2/ros2/issues/397
  (void) _init;
  strawberry_msgs__srv__CaptureSnapshot_Response__init(message_memory);
}

void strawberry_msgs__srv__CaptureSnapshot_Response__rosidl_typesupport_introspection_c__CaptureSnapshot_Response_fini_function(void * message_memory)
{
  strawberry_msgs__srv__CaptureSnapshot_Response__fini(message_memory);
}

static rosidl_typesupport_introspection_c__MessageMember strawberry_msgs__srv__CaptureSnapshot_Response__rosidl_typesupport_introspection_c__CaptureSnapshot_Response_message_member_array[4] = {
  {
    "success",  // name
    rosidl_typesupport_introspection_c__ROS_TYPE_BOOLEAN,  // type
    0,  // upper bound of string
    NULL,  // members of sub message
    false,  // is array
    0,  // array size
    false,  // is upper bound
    offsetof(strawberry_msgs__srv__CaptureSnapshot_Response, success),  // bytes offset in struct
    NULL,  // default value
    NULL,  // size() function pointer
    NULL,  // get_const(index) function pointer
    NULL,  // get(index) function pointer
    NULL,  // fetch(index, &value) function pointer
    NULL,  // assign(index, value) function pointer
    NULL  // resize(index) function pointer
  },
  {
    "message",  // name
    rosidl_typesupport_introspection_c__ROS_TYPE_STRING,  // type
    0,  // upper bound of string
    NULL,  // members of sub message
    false,  // is array
    0,  // array size
    false,  // is upper bound
    offsetof(strawberry_msgs__srv__CaptureSnapshot_Response, message),  // bytes offset in struct
    NULL,  // default value
    NULL,  // size() function pointer
    NULL,  // get_const(index) function pointer
    NULL,  // get(index) function pointer
    NULL,  // fetch(index, &value) function pointer
    NULL,  // assign(index, value) function pointer
    NULL  // resize(index) function pointer
  },
  {
    "rgb_path",  // name
    rosidl_typesupport_introspection_c__ROS_TYPE_STRING,  // type
    0,  // upper bound of string
    NULL,  // members of sub message
    false,  // is array
    0,  // array size
    false,  // is upper bound
    offsetof(strawberry_msgs__srv__CaptureSnapshot_Response, rgb_path),  // bytes offset in struct
    NULL,  // default value
    NULL,  // size() function pointer
    NULL,  // get_const(index) function pointer
    NULL,  // get(index) function pointer
    NULL,  // fetch(index, &value) function pointer
    NULL,  // assign(index, value) function pointer
    NULL  // resize(index) function pointer
  },
  {
    "depth_path",  // name
    rosidl_typesupport_introspection_c__ROS_TYPE_STRING,  // type
    0,  // upper bound of string
    NULL,  // members of sub message
    false,  // is array
    0,  // array size
    false,  // is upper bound
    offsetof(strawberry_msgs__srv__CaptureSnapshot_Response, depth_path),  // bytes offset in struct
    NULL,  // default value
    NULL,  // size() function pointer
    NULL,  // get_const(index) function pointer
    NULL,  // get(index) function pointer
    NULL,  // fetch(index, &value) function pointer
    NULL,  // assign(index, value) function pointer
    NULL  // resize(index) function pointer
  }
};

static const rosidl_typesupport_introspection_c__MessageMembers strawberry_msgs__srv__CaptureSnapshot_Response__rosidl_typesupport_introspection_c__CaptureSnapshot_Response_message_members = {
  "strawberry_msgs__srv",  // message namespace
  "CaptureSnapshot_Response",  // message name
  4,  // number of fields
  sizeof(strawberry_msgs__srv__CaptureSnapshot_Response),
  strawberry_msgs__srv__CaptureSnapshot_Response__rosidl_typesupport_introspection_c__CaptureSnapshot_Response_message_member_array,  // message members
  strawberry_msgs__srv__CaptureSnapshot_Response__rosidl_typesupport_introspection_c__CaptureSnapshot_Response_init_function,  // function to initialize message memory (memory has to be allocated)
  strawberry_msgs__srv__CaptureSnapshot_Response__rosidl_typesupport_introspection_c__CaptureSnapshot_Response_fini_function  // function to terminate message instance (will not free memory)
};

// this is not const since it must be initialized on first access
// since C does not allow non-integral compile-time constants
static rosidl_message_type_support_t strawberry_msgs__srv__CaptureSnapshot_Response__rosidl_typesupport_introspection_c__CaptureSnapshot_Response_message_type_support_handle = {
  0,
  &strawberry_msgs__srv__CaptureSnapshot_Response__rosidl_typesupport_introspection_c__CaptureSnapshot_Response_message_members,
  get_message_typesupport_handle_function,
};

ROSIDL_TYPESUPPORT_INTROSPECTION_C_EXPORT_strawberry_msgs
const rosidl_message_type_support_t *
ROSIDL_TYPESUPPORT_INTERFACE__MESSAGE_SYMBOL_NAME(rosidl_typesupport_introspection_c, strawberry_msgs, srv, CaptureSnapshot_Response)() {
  if (!strawberry_msgs__srv__CaptureSnapshot_Response__rosidl_typesupport_introspection_c__CaptureSnapshot_Response_message_type_support_handle.typesupport_identifier) {
    strawberry_msgs__srv__CaptureSnapshot_Response__rosidl_typesupport_introspection_c__CaptureSnapshot_Response_message_type_support_handle.typesupport_identifier =
      rosidl_typesupport_introspection_c__identifier;
  }
  return &strawberry_msgs__srv__CaptureSnapshot_Response__rosidl_typesupport_introspection_c__CaptureSnapshot_Response_message_type_support_handle;
}
#ifdef __cplusplus
}
#endif

#include "rosidl_runtime_c/service_type_support_struct.h"
// already included above
// #include "strawberry_msgs/msg/rosidl_typesupport_introspection_c__visibility_control.h"
// already included above
// #include "strawberry_msgs/srv/detail/capture_snapshot__rosidl_typesupport_introspection_c.h"
// already included above
// #include "rosidl_typesupport_introspection_c/identifier.h"
#include "rosidl_typesupport_introspection_c/service_introspection.h"

// this is intentionally not const to allow initialization later to prevent an initialization race
static rosidl_typesupport_introspection_c__ServiceMembers strawberry_msgs__srv__detail__capture_snapshot__rosidl_typesupport_introspection_c__CaptureSnapshot_service_members = {
  "strawberry_msgs__srv",  // service namespace
  "CaptureSnapshot",  // service name
  // these two fields are initialized below on the first access
  NULL,  // request message
  // strawberry_msgs__srv__detail__capture_snapshot__rosidl_typesupport_introspection_c__CaptureSnapshot_Request_message_type_support_handle,
  NULL  // response message
  // strawberry_msgs__srv__detail__capture_snapshot__rosidl_typesupport_introspection_c__CaptureSnapshot_Response_message_type_support_handle
};

static rosidl_service_type_support_t strawberry_msgs__srv__detail__capture_snapshot__rosidl_typesupport_introspection_c__CaptureSnapshot_service_type_support_handle = {
  0,
  &strawberry_msgs__srv__detail__capture_snapshot__rosidl_typesupport_introspection_c__CaptureSnapshot_service_members,
  get_service_typesupport_handle_function,
};

// Forward declaration of request/response type support functions
const rosidl_message_type_support_t *
ROSIDL_TYPESUPPORT_INTERFACE__MESSAGE_SYMBOL_NAME(rosidl_typesupport_introspection_c, strawberry_msgs, srv, CaptureSnapshot_Request)();

const rosidl_message_type_support_t *
ROSIDL_TYPESUPPORT_INTERFACE__MESSAGE_SYMBOL_NAME(rosidl_typesupport_introspection_c, strawberry_msgs, srv, CaptureSnapshot_Response)();

ROSIDL_TYPESUPPORT_INTROSPECTION_C_EXPORT_strawberry_msgs
const rosidl_service_type_support_t *
ROSIDL_TYPESUPPORT_INTERFACE__SERVICE_SYMBOL_NAME(rosidl_typesupport_introspection_c, strawberry_msgs, srv, CaptureSnapshot)() {
  if (!strawberry_msgs__srv__detail__capture_snapshot__rosidl_typesupport_introspection_c__CaptureSnapshot_service_type_support_handle.typesupport_identifier) {
    strawberry_msgs__srv__detail__capture_snapshot__rosidl_typesupport_introspection_c__CaptureSnapshot_service_type_support_handle.typesupport_identifier =
      rosidl_typesupport_introspection_c__identifier;
  }
  rosidl_typesupport_introspection_c__ServiceMembers * service_members =
    (rosidl_typesupport_introspection_c__ServiceMembers *)strawberry_msgs__srv__detail__capture_snapshot__rosidl_typesupport_introspection_c__CaptureSnapshot_service_type_support_handle.data;

  if (!service_members->request_members_) {
    service_members->request_members_ =
      (const rosidl_typesupport_introspection_c__MessageMembers *)
      ROSIDL_TYPESUPPORT_INTERFACE__MESSAGE_SYMBOL_NAME(rosidl_typesupport_introspection_c, strawberry_msgs, srv, CaptureSnapshot_Request)()->data;
  }
  if (!service_members->response_members_) {
    service_members->response_members_ =
      (const rosidl_typesupport_introspection_c__MessageMembers *)
      ROSIDL_TYPESUPPORT_INTERFACE__MESSAGE_SYMBOL_NAME(rosidl_typesupport_introspection_c, strawberry_msgs, srv, CaptureSnapshot_Response)()->data;
  }

  return &strawberry_msgs__srv__detail__capture_snapshot__rosidl_typesupport_introspection_c__CaptureSnapshot_service_type_support_handle;
}
