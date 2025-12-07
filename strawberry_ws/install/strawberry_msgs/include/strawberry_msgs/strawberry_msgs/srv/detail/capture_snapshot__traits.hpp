// generated from rosidl_generator_cpp/resource/idl__traits.hpp.em
// with input from strawberry_msgs:srv/CaptureSnapshot.idl
// generated code does not contain a copyright notice

#ifndef STRAWBERRY_MSGS__SRV__DETAIL__CAPTURE_SNAPSHOT__TRAITS_HPP_
#define STRAWBERRY_MSGS__SRV__DETAIL__CAPTURE_SNAPSHOT__TRAITS_HPP_

#include <stdint.h>

#include <sstream>
#include <string>
#include <type_traits>

#include "strawberry_msgs/srv/detail/capture_snapshot__struct.hpp"
#include "rosidl_runtime_cpp/traits.hpp"

namespace strawberry_msgs
{

namespace srv
{

inline void to_flow_style_yaml(
  const CaptureSnapshot_Request & msg,
  std::ostream & out)
{
  out << "{";
  // member: plant_id
  {
    out << "plant_id: ";
    rosidl_generator_traits::value_to_yaml(msg.plant_id, out);
    out << ", ";
  }

  // member: view_id
  {
    out << "view_id: ";
    rosidl_generator_traits::value_to_yaml(msg.view_id, out);
  }
  out << "}";
}  // NOLINT(readability/fn_size)

inline void to_block_style_yaml(
  const CaptureSnapshot_Request & msg,
  std::ostream & out, size_t indentation = 0)
{
  // member: plant_id
  {
    if (indentation > 0) {
      out << std::string(indentation, ' ');
    }
    out << "plant_id: ";
    rosidl_generator_traits::value_to_yaml(msg.plant_id, out);
    out << "\n";
  }

  // member: view_id
  {
    if (indentation > 0) {
      out << std::string(indentation, ' ');
    }
    out << "view_id: ";
    rosidl_generator_traits::value_to_yaml(msg.view_id, out);
    out << "\n";
  }
}  // NOLINT(readability/fn_size)

inline std::string to_yaml(const CaptureSnapshot_Request & msg, bool use_flow_style = false)
{
  std::ostringstream out;
  if (use_flow_style) {
    to_flow_style_yaml(msg, out);
  } else {
    to_block_style_yaml(msg, out);
  }
  return out.str();
}

}  // namespace srv

}  // namespace strawberry_msgs

namespace rosidl_generator_traits
{

[[deprecated("use strawberry_msgs::srv::to_block_style_yaml() instead")]]
inline void to_yaml(
  const strawberry_msgs::srv::CaptureSnapshot_Request & msg,
  std::ostream & out, size_t indentation = 0)
{
  strawberry_msgs::srv::to_block_style_yaml(msg, out, indentation);
}

[[deprecated("use strawberry_msgs::srv::to_yaml() instead")]]
inline std::string to_yaml(const strawberry_msgs::srv::CaptureSnapshot_Request & msg)
{
  return strawberry_msgs::srv::to_yaml(msg);
}

template<>
inline const char * data_type<strawberry_msgs::srv::CaptureSnapshot_Request>()
{
  return "strawberry_msgs::srv::CaptureSnapshot_Request";
}

template<>
inline const char * name<strawberry_msgs::srv::CaptureSnapshot_Request>()
{
  return "strawberry_msgs/srv/CaptureSnapshot_Request";
}

template<>
struct has_fixed_size<strawberry_msgs::srv::CaptureSnapshot_Request>
  : std::integral_constant<bool, true> {};

template<>
struct has_bounded_size<strawberry_msgs::srv::CaptureSnapshot_Request>
  : std::integral_constant<bool, true> {};

template<>
struct is_message<strawberry_msgs::srv::CaptureSnapshot_Request>
  : std::true_type {};

}  // namespace rosidl_generator_traits

namespace strawberry_msgs
{

namespace srv
{

inline void to_flow_style_yaml(
  const CaptureSnapshot_Response & msg,
  std::ostream & out)
{
  out << "{";
  // member: success
  {
    out << "success: ";
    rosidl_generator_traits::value_to_yaml(msg.success, out);
    out << ", ";
  }

  // member: message
  {
    out << "message: ";
    rosidl_generator_traits::value_to_yaml(msg.message, out);
    out << ", ";
  }

  // member: rgb_path
  {
    out << "rgb_path: ";
    rosidl_generator_traits::value_to_yaml(msg.rgb_path, out);
    out << ", ";
  }

  // member: depth_path
  {
    out << "depth_path: ";
    rosidl_generator_traits::value_to_yaml(msg.depth_path, out);
  }
  out << "}";
}  // NOLINT(readability/fn_size)

inline void to_block_style_yaml(
  const CaptureSnapshot_Response & msg,
  std::ostream & out, size_t indentation = 0)
{
  // member: success
  {
    if (indentation > 0) {
      out << std::string(indentation, ' ');
    }
    out << "success: ";
    rosidl_generator_traits::value_to_yaml(msg.success, out);
    out << "\n";
  }

  // member: message
  {
    if (indentation > 0) {
      out << std::string(indentation, ' ');
    }
    out << "message: ";
    rosidl_generator_traits::value_to_yaml(msg.message, out);
    out << "\n";
  }

  // member: rgb_path
  {
    if (indentation > 0) {
      out << std::string(indentation, ' ');
    }
    out << "rgb_path: ";
    rosidl_generator_traits::value_to_yaml(msg.rgb_path, out);
    out << "\n";
  }

  // member: depth_path
  {
    if (indentation > 0) {
      out << std::string(indentation, ' ');
    }
    out << "depth_path: ";
    rosidl_generator_traits::value_to_yaml(msg.depth_path, out);
    out << "\n";
  }
}  // NOLINT(readability/fn_size)

inline std::string to_yaml(const CaptureSnapshot_Response & msg, bool use_flow_style = false)
{
  std::ostringstream out;
  if (use_flow_style) {
    to_flow_style_yaml(msg, out);
  } else {
    to_block_style_yaml(msg, out);
  }
  return out.str();
}

}  // namespace srv

}  // namespace strawberry_msgs

namespace rosidl_generator_traits
{

[[deprecated("use strawberry_msgs::srv::to_block_style_yaml() instead")]]
inline void to_yaml(
  const strawberry_msgs::srv::CaptureSnapshot_Response & msg,
  std::ostream & out, size_t indentation = 0)
{
  strawberry_msgs::srv::to_block_style_yaml(msg, out, indentation);
}

[[deprecated("use strawberry_msgs::srv::to_yaml() instead")]]
inline std::string to_yaml(const strawberry_msgs::srv::CaptureSnapshot_Response & msg)
{
  return strawberry_msgs::srv::to_yaml(msg);
}

template<>
inline const char * data_type<strawberry_msgs::srv::CaptureSnapshot_Response>()
{
  return "strawberry_msgs::srv::CaptureSnapshot_Response";
}

template<>
inline const char * name<strawberry_msgs::srv::CaptureSnapshot_Response>()
{
  return "strawberry_msgs/srv/CaptureSnapshot_Response";
}

template<>
struct has_fixed_size<strawberry_msgs::srv::CaptureSnapshot_Response>
  : std::integral_constant<bool, false> {};

template<>
struct has_bounded_size<strawberry_msgs::srv::CaptureSnapshot_Response>
  : std::integral_constant<bool, false> {};

template<>
struct is_message<strawberry_msgs::srv::CaptureSnapshot_Response>
  : std::true_type {};

}  // namespace rosidl_generator_traits

namespace rosidl_generator_traits
{

template<>
inline const char * data_type<strawberry_msgs::srv::CaptureSnapshot>()
{
  return "strawberry_msgs::srv::CaptureSnapshot";
}

template<>
inline const char * name<strawberry_msgs::srv::CaptureSnapshot>()
{
  return "strawberry_msgs/srv/CaptureSnapshot";
}

template<>
struct has_fixed_size<strawberry_msgs::srv::CaptureSnapshot>
  : std::integral_constant<
    bool,
    has_fixed_size<strawberry_msgs::srv::CaptureSnapshot_Request>::value &&
    has_fixed_size<strawberry_msgs::srv::CaptureSnapshot_Response>::value
  >
{
};

template<>
struct has_bounded_size<strawberry_msgs::srv::CaptureSnapshot>
  : std::integral_constant<
    bool,
    has_bounded_size<strawberry_msgs::srv::CaptureSnapshot_Request>::value &&
    has_bounded_size<strawberry_msgs::srv::CaptureSnapshot_Response>::value
  >
{
};

template<>
struct is_service<strawberry_msgs::srv::CaptureSnapshot>
  : std::true_type
{
};

template<>
struct is_service_request<strawberry_msgs::srv::CaptureSnapshot_Request>
  : std::true_type
{
};

template<>
struct is_service_response<strawberry_msgs::srv::CaptureSnapshot_Response>
  : std::true_type
{
};

}  // namespace rosidl_generator_traits

#endif  // STRAWBERRY_MSGS__SRV__DETAIL__CAPTURE_SNAPSHOT__TRAITS_HPP_
