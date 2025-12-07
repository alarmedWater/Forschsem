// generated from rosidl_generator_c/resource/idl__functions.c.em
// with input from strawberry_msgs:srv/CaptureSnapshot.idl
// generated code does not contain a copyright notice
#include "strawberry_msgs/srv/detail/capture_snapshot__functions.h"

#include <assert.h>
#include <stdbool.h>
#include <stdlib.h>
#include <string.h>

#include "rcutils/allocator.h"

bool
strawberry_msgs__srv__CaptureSnapshot_Request__init(strawberry_msgs__srv__CaptureSnapshot_Request * msg)
{
  if (!msg) {
    return false;
  }
  // plant_id
  // view_id
  return true;
}

void
strawberry_msgs__srv__CaptureSnapshot_Request__fini(strawberry_msgs__srv__CaptureSnapshot_Request * msg)
{
  if (!msg) {
    return;
  }
  // plant_id
  // view_id
}

bool
strawberry_msgs__srv__CaptureSnapshot_Request__are_equal(const strawberry_msgs__srv__CaptureSnapshot_Request * lhs, const strawberry_msgs__srv__CaptureSnapshot_Request * rhs)
{
  if (!lhs || !rhs) {
    return false;
  }
  // plant_id
  if (lhs->plant_id != rhs->plant_id) {
    return false;
  }
  // view_id
  if (lhs->view_id != rhs->view_id) {
    return false;
  }
  return true;
}

bool
strawberry_msgs__srv__CaptureSnapshot_Request__copy(
  const strawberry_msgs__srv__CaptureSnapshot_Request * input,
  strawberry_msgs__srv__CaptureSnapshot_Request * output)
{
  if (!input || !output) {
    return false;
  }
  // plant_id
  output->plant_id = input->plant_id;
  // view_id
  output->view_id = input->view_id;
  return true;
}

strawberry_msgs__srv__CaptureSnapshot_Request *
strawberry_msgs__srv__CaptureSnapshot_Request__create()
{
  rcutils_allocator_t allocator = rcutils_get_default_allocator();
  strawberry_msgs__srv__CaptureSnapshot_Request * msg = (strawberry_msgs__srv__CaptureSnapshot_Request *)allocator.allocate(sizeof(strawberry_msgs__srv__CaptureSnapshot_Request), allocator.state);
  if (!msg) {
    return NULL;
  }
  memset(msg, 0, sizeof(strawberry_msgs__srv__CaptureSnapshot_Request));
  bool success = strawberry_msgs__srv__CaptureSnapshot_Request__init(msg);
  if (!success) {
    allocator.deallocate(msg, allocator.state);
    return NULL;
  }
  return msg;
}

void
strawberry_msgs__srv__CaptureSnapshot_Request__destroy(strawberry_msgs__srv__CaptureSnapshot_Request * msg)
{
  rcutils_allocator_t allocator = rcutils_get_default_allocator();
  if (msg) {
    strawberry_msgs__srv__CaptureSnapshot_Request__fini(msg);
  }
  allocator.deallocate(msg, allocator.state);
}


bool
strawberry_msgs__srv__CaptureSnapshot_Request__Sequence__init(strawberry_msgs__srv__CaptureSnapshot_Request__Sequence * array, size_t size)
{
  if (!array) {
    return false;
  }
  rcutils_allocator_t allocator = rcutils_get_default_allocator();
  strawberry_msgs__srv__CaptureSnapshot_Request * data = NULL;

  if (size) {
    data = (strawberry_msgs__srv__CaptureSnapshot_Request *)allocator.zero_allocate(size, sizeof(strawberry_msgs__srv__CaptureSnapshot_Request), allocator.state);
    if (!data) {
      return false;
    }
    // initialize all array elements
    size_t i;
    for (i = 0; i < size; ++i) {
      bool success = strawberry_msgs__srv__CaptureSnapshot_Request__init(&data[i]);
      if (!success) {
        break;
      }
    }
    if (i < size) {
      // if initialization failed finalize the already initialized array elements
      for (; i > 0; --i) {
        strawberry_msgs__srv__CaptureSnapshot_Request__fini(&data[i - 1]);
      }
      allocator.deallocate(data, allocator.state);
      return false;
    }
  }
  array->data = data;
  array->size = size;
  array->capacity = size;
  return true;
}

void
strawberry_msgs__srv__CaptureSnapshot_Request__Sequence__fini(strawberry_msgs__srv__CaptureSnapshot_Request__Sequence * array)
{
  if (!array) {
    return;
  }
  rcutils_allocator_t allocator = rcutils_get_default_allocator();

  if (array->data) {
    // ensure that data and capacity values are consistent
    assert(array->capacity > 0);
    // finalize all array elements
    for (size_t i = 0; i < array->capacity; ++i) {
      strawberry_msgs__srv__CaptureSnapshot_Request__fini(&array->data[i]);
    }
    allocator.deallocate(array->data, allocator.state);
    array->data = NULL;
    array->size = 0;
    array->capacity = 0;
  } else {
    // ensure that data, size, and capacity values are consistent
    assert(0 == array->size);
    assert(0 == array->capacity);
  }
}

strawberry_msgs__srv__CaptureSnapshot_Request__Sequence *
strawberry_msgs__srv__CaptureSnapshot_Request__Sequence__create(size_t size)
{
  rcutils_allocator_t allocator = rcutils_get_default_allocator();
  strawberry_msgs__srv__CaptureSnapshot_Request__Sequence * array = (strawberry_msgs__srv__CaptureSnapshot_Request__Sequence *)allocator.allocate(sizeof(strawberry_msgs__srv__CaptureSnapshot_Request__Sequence), allocator.state);
  if (!array) {
    return NULL;
  }
  bool success = strawberry_msgs__srv__CaptureSnapshot_Request__Sequence__init(array, size);
  if (!success) {
    allocator.deallocate(array, allocator.state);
    return NULL;
  }
  return array;
}

void
strawberry_msgs__srv__CaptureSnapshot_Request__Sequence__destroy(strawberry_msgs__srv__CaptureSnapshot_Request__Sequence * array)
{
  rcutils_allocator_t allocator = rcutils_get_default_allocator();
  if (array) {
    strawberry_msgs__srv__CaptureSnapshot_Request__Sequence__fini(array);
  }
  allocator.deallocate(array, allocator.state);
}

bool
strawberry_msgs__srv__CaptureSnapshot_Request__Sequence__are_equal(const strawberry_msgs__srv__CaptureSnapshot_Request__Sequence * lhs, const strawberry_msgs__srv__CaptureSnapshot_Request__Sequence * rhs)
{
  if (!lhs || !rhs) {
    return false;
  }
  if (lhs->size != rhs->size) {
    return false;
  }
  for (size_t i = 0; i < lhs->size; ++i) {
    if (!strawberry_msgs__srv__CaptureSnapshot_Request__are_equal(&(lhs->data[i]), &(rhs->data[i]))) {
      return false;
    }
  }
  return true;
}

bool
strawberry_msgs__srv__CaptureSnapshot_Request__Sequence__copy(
  const strawberry_msgs__srv__CaptureSnapshot_Request__Sequence * input,
  strawberry_msgs__srv__CaptureSnapshot_Request__Sequence * output)
{
  if (!input || !output) {
    return false;
  }
  if (output->capacity < input->size) {
    const size_t allocation_size =
      input->size * sizeof(strawberry_msgs__srv__CaptureSnapshot_Request);
    rcutils_allocator_t allocator = rcutils_get_default_allocator();
    strawberry_msgs__srv__CaptureSnapshot_Request * data =
      (strawberry_msgs__srv__CaptureSnapshot_Request *)allocator.reallocate(
      output->data, allocation_size, allocator.state);
    if (!data) {
      return false;
    }
    // If reallocation succeeded, memory may or may not have been moved
    // to fulfill the allocation request, invalidating output->data.
    output->data = data;
    for (size_t i = output->capacity; i < input->size; ++i) {
      if (!strawberry_msgs__srv__CaptureSnapshot_Request__init(&output->data[i])) {
        // If initialization of any new item fails, roll back
        // all previously initialized items. Existing items
        // in output are to be left unmodified.
        for (; i-- > output->capacity; ) {
          strawberry_msgs__srv__CaptureSnapshot_Request__fini(&output->data[i]);
        }
        return false;
      }
    }
    output->capacity = input->size;
  }
  output->size = input->size;
  for (size_t i = 0; i < input->size; ++i) {
    if (!strawberry_msgs__srv__CaptureSnapshot_Request__copy(
        &(input->data[i]), &(output->data[i])))
    {
      return false;
    }
  }
  return true;
}


// Include directives for member types
// Member `message`
// Member `rgb_path`
// Member `depth_path`
#include "rosidl_runtime_c/string_functions.h"

bool
strawberry_msgs__srv__CaptureSnapshot_Response__init(strawberry_msgs__srv__CaptureSnapshot_Response * msg)
{
  if (!msg) {
    return false;
  }
  // success
  // message
  if (!rosidl_runtime_c__String__init(&msg->message)) {
    strawberry_msgs__srv__CaptureSnapshot_Response__fini(msg);
    return false;
  }
  // rgb_path
  if (!rosidl_runtime_c__String__init(&msg->rgb_path)) {
    strawberry_msgs__srv__CaptureSnapshot_Response__fini(msg);
    return false;
  }
  // depth_path
  if (!rosidl_runtime_c__String__init(&msg->depth_path)) {
    strawberry_msgs__srv__CaptureSnapshot_Response__fini(msg);
    return false;
  }
  return true;
}

void
strawberry_msgs__srv__CaptureSnapshot_Response__fini(strawberry_msgs__srv__CaptureSnapshot_Response * msg)
{
  if (!msg) {
    return;
  }
  // success
  // message
  rosidl_runtime_c__String__fini(&msg->message);
  // rgb_path
  rosidl_runtime_c__String__fini(&msg->rgb_path);
  // depth_path
  rosidl_runtime_c__String__fini(&msg->depth_path);
}

bool
strawberry_msgs__srv__CaptureSnapshot_Response__are_equal(const strawberry_msgs__srv__CaptureSnapshot_Response * lhs, const strawberry_msgs__srv__CaptureSnapshot_Response * rhs)
{
  if (!lhs || !rhs) {
    return false;
  }
  // success
  if (lhs->success != rhs->success) {
    return false;
  }
  // message
  if (!rosidl_runtime_c__String__are_equal(
      &(lhs->message), &(rhs->message)))
  {
    return false;
  }
  // rgb_path
  if (!rosidl_runtime_c__String__are_equal(
      &(lhs->rgb_path), &(rhs->rgb_path)))
  {
    return false;
  }
  // depth_path
  if (!rosidl_runtime_c__String__are_equal(
      &(lhs->depth_path), &(rhs->depth_path)))
  {
    return false;
  }
  return true;
}

bool
strawberry_msgs__srv__CaptureSnapshot_Response__copy(
  const strawberry_msgs__srv__CaptureSnapshot_Response * input,
  strawberry_msgs__srv__CaptureSnapshot_Response * output)
{
  if (!input || !output) {
    return false;
  }
  // success
  output->success = input->success;
  // message
  if (!rosidl_runtime_c__String__copy(
      &(input->message), &(output->message)))
  {
    return false;
  }
  // rgb_path
  if (!rosidl_runtime_c__String__copy(
      &(input->rgb_path), &(output->rgb_path)))
  {
    return false;
  }
  // depth_path
  if (!rosidl_runtime_c__String__copy(
      &(input->depth_path), &(output->depth_path)))
  {
    return false;
  }
  return true;
}

strawberry_msgs__srv__CaptureSnapshot_Response *
strawberry_msgs__srv__CaptureSnapshot_Response__create()
{
  rcutils_allocator_t allocator = rcutils_get_default_allocator();
  strawberry_msgs__srv__CaptureSnapshot_Response * msg = (strawberry_msgs__srv__CaptureSnapshot_Response *)allocator.allocate(sizeof(strawberry_msgs__srv__CaptureSnapshot_Response), allocator.state);
  if (!msg) {
    return NULL;
  }
  memset(msg, 0, sizeof(strawberry_msgs__srv__CaptureSnapshot_Response));
  bool success = strawberry_msgs__srv__CaptureSnapshot_Response__init(msg);
  if (!success) {
    allocator.deallocate(msg, allocator.state);
    return NULL;
  }
  return msg;
}

void
strawberry_msgs__srv__CaptureSnapshot_Response__destroy(strawberry_msgs__srv__CaptureSnapshot_Response * msg)
{
  rcutils_allocator_t allocator = rcutils_get_default_allocator();
  if (msg) {
    strawberry_msgs__srv__CaptureSnapshot_Response__fini(msg);
  }
  allocator.deallocate(msg, allocator.state);
}


bool
strawberry_msgs__srv__CaptureSnapshot_Response__Sequence__init(strawberry_msgs__srv__CaptureSnapshot_Response__Sequence * array, size_t size)
{
  if (!array) {
    return false;
  }
  rcutils_allocator_t allocator = rcutils_get_default_allocator();
  strawberry_msgs__srv__CaptureSnapshot_Response * data = NULL;

  if (size) {
    data = (strawberry_msgs__srv__CaptureSnapshot_Response *)allocator.zero_allocate(size, sizeof(strawberry_msgs__srv__CaptureSnapshot_Response), allocator.state);
    if (!data) {
      return false;
    }
    // initialize all array elements
    size_t i;
    for (i = 0; i < size; ++i) {
      bool success = strawberry_msgs__srv__CaptureSnapshot_Response__init(&data[i]);
      if (!success) {
        break;
      }
    }
    if (i < size) {
      // if initialization failed finalize the already initialized array elements
      for (; i > 0; --i) {
        strawberry_msgs__srv__CaptureSnapshot_Response__fini(&data[i - 1]);
      }
      allocator.deallocate(data, allocator.state);
      return false;
    }
  }
  array->data = data;
  array->size = size;
  array->capacity = size;
  return true;
}

void
strawberry_msgs__srv__CaptureSnapshot_Response__Sequence__fini(strawberry_msgs__srv__CaptureSnapshot_Response__Sequence * array)
{
  if (!array) {
    return;
  }
  rcutils_allocator_t allocator = rcutils_get_default_allocator();

  if (array->data) {
    // ensure that data and capacity values are consistent
    assert(array->capacity > 0);
    // finalize all array elements
    for (size_t i = 0; i < array->capacity; ++i) {
      strawberry_msgs__srv__CaptureSnapshot_Response__fini(&array->data[i]);
    }
    allocator.deallocate(array->data, allocator.state);
    array->data = NULL;
    array->size = 0;
    array->capacity = 0;
  } else {
    // ensure that data, size, and capacity values are consistent
    assert(0 == array->size);
    assert(0 == array->capacity);
  }
}

strawberry_msgs__srv__CaptureSnapshot_Response__Sequence *
strawberry_msgs__srv__CaptureSnapshot_Response__Sequence__create(size_t size)
{
  rcutils_allocator_t allocator = rcutils_get_default_allocator();
  strawberry_msgs__srv__CaptureSnapshot_Response__Sequence * array = (strawberry_msgs__srv__CaptureSnapshot_Response__Sequence *)allocator.allocate(sizeof(strawberry_msgs__srv__CaptureSnapshot_Response__Sequence), allocator.state);
  if (!array) {
    return NULL;
  }
  bool success = strawberry_msgs__srv__CaptureSnapshot_Response__Sequence__init(array, size);
  if (!success) {
    allocator.deallocate(array, allocator.state);
    return NULL;
  }
  return array;
}

void
strawberry_msgs__srv__CaptureSnapshot_Response__Sequence__destroy(strawberry_msgs__srv__CaptureSnapshot_Response__Sequence * array)
{
  rcutils_allocator_t allocator = rcutils_get_default_allocator();
  if (array) {
    strawberry_msgs__srv__CaptureSnapshot_Response__Sequence__fini(array);
  }
  allocator.deallocate(array, allocator.state);
}

bool
strawberry_msgs__srv__CaptureSnapshot_Response__Sequence__are_equal(const strawberry_msgs__srv__CaptureSnapshot_Response__Sequence * lhs, const strawberry_msgs__srv__CaptureSnapshot_Response__Sequence * rhs)
{
  if (!lhs || !rhs) {
    return false;
  }
  if (lhs->size != rhs->size) {
    return false;
  }
  for (size_t i = 0; i < lhs->size; ++i) {
    if (!strawberry_msgs__srv__CaptureSnapshot_Response__are_equal(&(lhs->data[i]), &(rhs->data[i]))) {
      return false;
    }
  }
  return true;
}

bool
strawberry_msgs__srv__CaptureSnapshot_Response__Sequence__copy(
  const strawberry_msgs__srv__CaptureSnapshot_Response__Sequence * input,
  strawberry_msgs__srv__CaptureSnapshot_Response__Sequence * output)
{
  if (!input || !output) {
    return false;
  }
  if (output->capacity < input->size) {
    const size_t allocation_size =
      input->size * sizeof(strawberry_msgs__srv__CaptureSnapshot_Response);
    rcutils_allocator_t allocator = rcutils_get_default_allocator();
    strawberry_msgs__srv__CaptureSnapshot_Response * data =
      (strawberry_msgs__srv__CaptureSnapshot_Response *)allocator.reallocate(
      output->data, allocation_size, allocator.state);
    if (!data) {
      return false;
    }
    // If reallocation succeeded, memory may or may not have been moved
    // to fulfill the allocation request, invalidating output->data.
    output->data = data;
    for (size_t i = output->capacity; i < input->size; ++i) {
      if (!strawberry_msgs__srv__CaptureSnapshot_Response__init(&output->data[i])) {
        // If initialization of any new item fails, roll back
        // all previously initialized items. Existing items
        // in output are to be left unmodified.
        for (; i-- > output->capacity; ) {
          strawberry_msgs__srv__CaptureSnapshot_Response__fini(&output->data[i]);
        }
        return false;
      }
    }
    output->capacity = input->size;
  }
  output->size = input->size;
  for (size_t i = 0; i < input->size; ++i) {
    if (!strawberry_msgs__srv__CaptureSnapshot_Response__copy(
        &(input->data[i]), &(output->data[i])))
    {
      return false;
    }
  }
  return true;
}
