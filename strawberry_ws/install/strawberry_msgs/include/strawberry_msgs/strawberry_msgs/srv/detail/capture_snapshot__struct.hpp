// generated from rosidl_generator_cpp/resource/idl__struct.hpp.em
// with input from strawberry_msgs:srv/CaptureSnapshot.idl
// generated code does not contain a copyright notice

#ifndef STRAWBERRY_MSGS__SRV__DETAIL__CAPTURE_SNAPSHOT__STRUCT_HPP_
#define STRAWBERRY_MSGS__SRV__DETAIL__CAPTURE_SNAPSHOT__STRUCT_HPP_

#include <algorithm>
#include <array>
#include <memory>
#include <string>
#include <vector>

#include "rosidl_runtime_cpp/bounded_vector.hpp"
#include "rosidl_runtime_cpp/message_initialization.hpp"


#ifndef _WIN32
# define DEPRECATED__strawberry_msgs__srv__CaptureSnapshot_Request __attribute__((deprecated))
#else
# define DEPRECATED__strawberry_msgs__srv__CaptureSnapshot_Request __declspec(deprecated)
#endif

namespace strawberry_msgs
{

namespace srv
{

// message struct
template<class ContainerAllocator>
struct CaptureSnapshot_Request_
{
  using Type = CaptureSnapshot_Request_<ContainerAllocator>;

  explicit CaptureSnapshot_Request_(rosidl_runtime_cpp::MessageInitialization _init = rosidl_runtime_cpp::MessageInitialization::ALL)
  {
    if (rosidl_runtime_cpp::MessageInitialization::ALL == _init ||
      rosidl_runtime_cpp::MessageInitialization::ZERO == _init)
    {
      this->plant_id = 0l;
      this->view_id = 0l;
    }
  }

  explicit CaptureSnapshot_Request_(const ContainerAllocator & _alloc, rosidl_runtime_cpp::MessageInitialization _init = rosidl_runtime_cpp::MessageInitialization::ALL)
  {
    (void)_alloc;
    if (rosidl_runtime_cpp::MessageInitialization::ALL == _init ||
      rosidl_runtime_cpp::MessageInitialization::ZERO == _init)
    {
      this->plant_id = 0l;
      this->view_id = 0l;
    }
  }

  // field types and members
  using _plant_id_type =
    int32_t;
  _plant_id_type plant_id;
  using _view_id_type =
    int32_t;
  _view_id_type view_id;

  // setters for named parameter idiom
  Type & set__plant_id(
    const int32_t & _arg)
  {
    this->plant_id = _arg;
    return *this;
  }
  Type & set__view_id(
    const int32_t & _arg)
  {
    this->view_id = _arg;
    return *this;
  }

  // constant declarations

  // pointer types
  using RawPtr =
    strawberry_msgs::srv::CaptureSnapshot_Request_<ContainerAllocator> *;
  using ConstRawPtr =
    const strawberry_msgs::srv::CaptureSnapshot_Request_<ContainerAllocator> *;
  using SharedPtr =
    std::shared_ptr<strawberry_msgs::srv::CaptureSnapshot_Request_<ContainerAllocator>>;
  using ConstSharedPtr =
    std::shared_ptr<strawberry_msgs::srv::CaptureSnapshot_Request_<ContainerAllocator> const>;

  template<typename Deleter = std::default_delete<
      strawberry_msgs::srv::CaptureSnapshot_Request_<ContainerAllocator>>>
  using UniquePtrWithDeleter =
    std::unique_ptr<strawberry_msgs::srv::CaptureSnapshot_Request_<ContainerAllocator>, Deleter>;

  using UniquePtr = UniquePtrWithDeleter<>;

  template<typename Deleter = std::default_delete<
      strawberry_msgs::srv::CaptureSnapshot_Request_<ContainerAllocator>>>
  using ConstUniquePtrWithDeleter =
    std::unique_ptr<strawberry_msgs::srv::CaptureSnapshot_Request_<ContainerAllocator> const, Deleter>;
  using ConstUniquePtr = ConstUniquePtrWithDeleter<>;

  using WeakPtr =
    std::weak_ptr<strawberry_msgs::srv::CaptureSnapshot_Request_<ContainerAllocator>>;
  using ConstWeakPtr =
    std::weak_ptr<strawberry_msgs::srv::CaptureSnapshot_Request_<ContainerAllocator> const>;

  // pointer types similar to ROS 1, use SharedPtr / ConstSharedPtr instead
  // NOTE: Can't use 'using' here because GNU C++ can't parse attributes properly
  typedef DEPRECATED__strawberry_msgs__srv__CaptureSnapshot_Request
    std::shared_ptr<strawberry_msgs::srv::CaptureSnapshot_Request_<ContainerAllocator>>
    Ptr;
  typedef DEPRECATED__strawberry_msgs__srv__CaptureSnapshot_Request
    std::shared_ptr<strawberry_msgs::srv::CaptureSnapshot_Request_<ContainerAllocator> const>
    ConstPtr;

  // comparison operators
  bool operator==(const CaptureSnapshot_Request_ & other) const
  {
    if (this->plant_id != other.plant_id) {
      return false;
    }
    if (this->view_id != other.view_id) {
      return false;
    }
    return true;
  }
  bool operator!=(const CaptureSnapshot_Request_ & other) const
  {
    return !this->operator==(other);
  }
};  // struct CaptureSnapshot_Request_

// alias to use template instance with default allocator
using CaptureSnapshot_Request =
  strawberry_msgs::srv::CaptureSnapshot_Request_<std::allocator<void>>;

// constant definitions

}  // namespace srv

}  // namespace strawberry_msgs


#ifndef _WIN32
# define DEPRECATED__strawberry_msgs__srv__CaptureSnapshot_Response __attribute__((deprecated))
#else
# define DEPRECATED__strawberry_msgs__srv__CaptureSnapshot_Response __declspec(deprecated)
#endif

namespace strawberry_msgs
{

namespace srv
{

// message struct
template<class ContainerAllocator>
struct CaptureSnapshot_Response_
{
  using Type = CaptureSnapshot_Response_<ContainerAllocator>;

  explicit CaptureSnapshot_Response_(rosidl_runtime_cpp::MessageInitialization _init = rosidl_runtime_cpp::MessageInitialization::ALL)
  {
    if (rosidl_runtime_cpp::MessageInitialization::ALL == _init ||
      rosidl_runtime_cpp::MessageInitialization::ZERO == _init)
    {
      this->success = false;
      this->message = "";
      this->rgb_path = "";
      this->depth_path = "";
    }
  }

  explicit CaptureSnapshot_Response_(const ContainerAllocator & _alloc, rosidl_runtime_cpp::MessageInitialization _init = rosidl_runtime_cpp::MessageInitialization::ALL)
  : message(_alloc),
    rgb_path(_alloc),
    depth_path(_alloc)
  {
    if (rosidl_runtime_cpp::MessageInitialization::ALL == _init ||
      rosidl_runtime_cpp::MessageInitialization::ZERO == _init)
    {
      this->success = false;
      this->message = "";
      this->rgb_path = "";
      this->depth_path = "";
    }
  }

  // field types and members
  using _success_type =
    bool;
  _success_type success;
  using _message_type =
    std::basic_string<char, std::char_traits<char>, typename std::allocator_traits<ContainerAllocator>::template rebind_alloc<char>>;
  _message_type message;
  using _rgb_path_type =
    std::basic_string<char, std::char_traits<char>, typename std::allocator_traits<ContainerAllocator>::template rebind_alloc<char>>;
  _rgb_path_type rgb_path;
  using _depth_path_type =
    std::basic_string<char, std::char_traits<char>, typename std::allocator_traits<ContainerAllocator>::template rebind_alloc<char>>;
  _depth_path_type depth_path;

  // setters for named parameter idiom
  Type & set__success(
    const bool & _arg)
  {
    this->success = _arg;
    return *this;
  }
  Type & set__message(
    const std::basic_string<char, std::char_traits<char>, typename std::allocator_traits<ContainerAllocator>::template rebind_alloc<char>> & _arg)
  {
    this->message = _arg;
    return *this;
  }
  Type & set__rgb_path(
    const std::basic_string<char, std::char_traits<char>, typename std::allocator_traits<ContainerAllocator>::template rebind_alloc<char>> & _arg)
  {
    this->rgb_path = _arg;
    return *this;
  }
  Type & set__depth_path(
    const std::basic_string<char, std::char_traits<char>, typename std::allocator_traits<ContainerAllocator>::template rebind_alloc<char>> & _arg)
  {
    this->depth_path = _arg;
    return *this;
  }

  // constant declarations

  // pointer types
  using RawPtr =
    strawberry_msgs::srv::CaptureSnapshot_Response_<ContainerAllocator> *;
  using ConstRawPtr =
    const strawberry_msgs::srv::CaptureSnapshot_Response_<ContainerAllocator> *;
  using SharedPtr =
    std::shared_ptr<strawberry_msgs::srv::CaptureSnapshot_Response_<ContainerAllocator>>;
  using ConstSharedPtr =
    std::shared_ptr<strawberry_msgs::srv::CaptureSnapshot_Response_<ContainerAllocator> const>;

  template<typename Deleter = std::default_delete<
      strawberry_msgs::srv::CaptureSnapshot_Response_<ContainerAllocator>>>
  using UniquePtrWithDeleter =
    std::unique_ptr<strawberry_msgs::srv::CaptureSnapshot_Response_<ContainerAllocator>, Deleter>;

  using UniquePtr = UniquePtrWithDeleter<>;

  template<typename Deleter = std::default_delete<
      strawberry_msgs::srv::CaptureSnapshot_Response_<ContainerAllocator>>>
  using ConstUniquePtrWithDeleter =
    std::unique_ptr<strawberry_msgs::srv::CaptureSnapshot_Response_<ContainerAllocator> const, Deleter>;
  using ConstUniquePtr = ConstUniquePtrWithDeleter<>;

  using WeakPtr =
    std::weak_ptr<strawberry_msgs::srv::CaptureSnapshot_Response_<ContainerAllocator>>;
  using ConstWeakPtr =
    std::weak_ptr<strawberry_msgs::srv::CaptureSnapshot_Response_<ContainerAllocator> const>;

  // pointer types similar to ROS 1, use SharedPtr / ConstSharedPtr instead
  // NOTE: Can't use 'using' here because GNU C++ can't parse attributes properly
  typedef DEPRECATED__strawberry_msgs__srv__CaptureSnapshot_Response
    std::shared_ptr<strawberry_msgs::srv::CaptureSnapshot_Response_<ContainerAllocator>>
    Ptr;
  typedef DEPRECATED__strawberry_msgs__srv__CaptureSnapshot_Response
    std::shared_ptr<strawberry_msgs::srv::CaptureSnapshot_Response_<ContainerAllocator> const>
    ConstPtr;

  // comparison operators
  bool operator==(const CaptureSnapshot_Response_ & other) const
  {
    if (this->success != other.success) {
      return false;
    }
    if (this->message != other.message) {
      return false;
    }
    if (this->rgb_path != other.rgb_path) {
      return false;
    }
    if (this->depth_path != other.depth_path) {
      return false;
    }
    return true;
  }
  bool operator!=(const CaptureSnapshot_Response_ & other) const
  {
    return !this->operator==(other);
  }
};  // struct CaptureSnapshot_Response_

// alias to use template instance with default allocator
using CaptureSnapshot_Response =
  strawberry_msgs::srv::CaptureSnapshot_Response_<std::allocator<void>>;

// constant definitions

}  // namespace srv

}  // namespace strawberry_msgs

namespace strawberry_msgs
{

namespace srv
{

struct CaptureSnapshot
{
  using Request = strawberry_msgs::srv::CaptureSnapshot_Request;
  using Response = strawberry_msgs::srv::CaptureSnapshot_Response;
};

}  // namespace srv

}  // namespace strawberry_msgs

#endif  // STRAWBERRY_MSGS__SRV__DETAIL__CAPTURE_SNAPSHOT__STRUCT_HPP_
