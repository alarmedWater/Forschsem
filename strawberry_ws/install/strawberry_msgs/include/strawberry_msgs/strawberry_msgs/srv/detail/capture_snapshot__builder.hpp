// generated from rosidl_generator_cpp/resource/idl__builder.hpp.em
// with input from strawberry_msgs:srv/CaptureSnapshot.idl
// generated code does not contain a copyright notice

#ifndef STRAWBERRY_MSGS__SRV__DETAIL__CAPTURE_SNAPSHOT__BUILDER_HPP_
#define STRAWBERRY_MSGS__SRV__DETAIL__CAPTURE_SNAPSHOT__BUILDER_HPP_

#include <algorithm>
#include <utility>

#include "strawberry_msgs/srv/detail/capture_snapshot__struct.hpp"
#include "rosidl_runtime_cpp/message_initialization.hpp"


namespace strawberry_msgs
{

namespace srv
{

namespace builder
{

class Init_CaptureSnapshot_Request_view_id
{
public:
  explicit Init_CaptureSnapshot_Request_view_id(::strawberry_msgs::srv::CaptureSnapshot_Request & msg)
  : msg_(msg)
  {}
  ::strawberry_msgs::srv::CaptureSnapshot_Request view_id(::strawberry_msgs::srv::CaptureSnapshot_Request::_view_id_type arg)
  {
    msg_.view_id = std::move(arg);
    return std::move(msg_);
  }

private:
  ::strawberry_msgs::srv::CaptureSnapshot_Request msg_;
};

class Init_CaptureSnapshot_Request_plant_id
{
public:
  Init_CaptureSnapshot_Request_plant_id()
  : msg_(::rosidl_runtime_cpp::MessageInitialization::SKIP)
  {}
  Init_CaptureSnapshot_Request_view_id plant_id(::strawberry_msgs::srv::CaptureSnapshot_Request::_plant_id_type arg)
  {
    msg_.plant_id = std::move(arg);
    return Init_CaptureSnapshot_Request_view_id(msg_);
  }

private:
  ::strawberry_msgs::srv::CaptureSnapshot_Request msg_;
};

}  // namespace builder

}  // namespace srv

template<typename MessageType>
auto build();

template<>
inline
auto build<::strawberry_msgs::srv::CaptureSnapshot_Request>()
{
  return strawberry_msgs::srv::builder::Init_CaptureSnapshot_Request_plant_id();
}

}  // namespace strawberry_msgs


namespace strawberry_msgs
{

namespace srv
{

namespace builder
{

class Init_CaptureSnapshot_Response_depth_path
{
public:
  explicit Init_CaptureSnapshot_Response_depth_path(::strawberry_msgs::srv::CaptureSnapshot_Response & msg)
  : msg_(msg)
  {}
  ::strawberry_msgs::srv::CaptureSnapshot_Response depth_path(::strawberry_msgs::srv::CaptureSnapshot_Response::_depth_path_type arg)
  {
    msg_.depth_path = std::move(arg);
    return std::move(msg_);
  }

private:
  ::strawberry_msgs::srv::CaptureSnapshot_Response msg_;
};

class Init_CaptureSnapshot_Response_rgb_path
{
public:
  explicit Init_CaptureSnapshot_Response_rgb_path(::strawberry_msgs::srv::CaptureSnapshot_Response & msg)
  : msg_(msg)
  {}
  Init_CaptureSnapshot_Response_depth_path rgb_path(::strawberry_msgs::srv::CaptureSnapshot_Response::_rgb_path_type arg)
  {
    msg_.rgb_path = std::move(arg);
    return Init_CaptureSnapshot_Response_depth_path(msg_);
  }

private:
  ::strawberry_msgs::srv::CaptureSnapshot_Response msg_;
};

class Init_CaptureSnapshot_Response_message
{
public:
  explicit Init_CaptureSnapshot_Response_message(::strawberry_msgs::srv::CaptureSnapshot_Response & msg)
  : msg_(msg)
  {}
  Init_CaptureSnapshot_Response_rgb_path message(::strawberry_msgs::srv::CaptureSnapshot_Response::_message_type arg)
  {
    msg_.message = std::move(arg);
    return Init_CaptureSnapshot_Response_rgb_path(msg_);
  }

private:
  ::strawberry_msgs::srv::CaptureSnapshot_Response msg_;
};

class Init_CaptureSnapshot_Response_success
{
public:
  Init_CaptureSnapshot_Response_success()
  : msg_(::rosidl_runtime_cpp::MessageInitialization::SKIP)
  {}
  Init_CaptureSnapshot_Response_message success(::strawberry_msgs::srv::CaptureSnapshot_Response::_success_type arg)
  {
    msg_.success = std::move(arg);
    return Init_CaptureSnapshot_Response_message(msg_);
  }

private:
  ::strawberry_msgs::srv::CaptureSnapshot_Response msg_;
};

}  // namespace builder

}  // namespace srv

template<typename MessageType>
auto build();

template<>
inline
auto build<::strawberry_msgs::srv::CaptureSnapshot_Response>()
{
  return strawberry_msgs::srv::builder::Init_CaptureSnapshot_Response_success();
}

}  // namespace strawberry_msgs

#endif  // STRAWBERRY_MSGS__SRV__DETAIL__CAPTURE_SNAPSHOT__BUILDER_HPP_
