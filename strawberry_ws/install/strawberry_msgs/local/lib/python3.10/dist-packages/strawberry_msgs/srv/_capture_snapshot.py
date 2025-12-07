# generated from rosidl_generator_py/resource/_idl.py.em
# with input from strawberry_msgs:srv/CaptureSnapshot.idl
# generated code does not contain a copyright notice


# Import statements for member types

import builtins  # noqa: E402, I100

import rosidl_parser.definition  # noqa: E402, I100


class Metaclass_CaptureSnapshot_Request(type):
    """Metaclass of message 'CaptureSnapshot_Request'."""

    _CREATE_ROS_MESSAGE = None
    _CONVERT_FROM_PY = None
    _CONVERT_TO_PY = None
    _DESTROY_ROS_MESSAGE = None
    _TYPE_SUPPORT = None

    __constants = {
    }

    @classmethod
    def __import_type_support__(cls):
        try:
            from rosidl_generator_py import import_type_support
            module = import_type_support('strawberry_msgs')
        except ImportError:
            import logging
            import traceback
            logger = logging.getLogger(
                'strawberry_msgs.srv.CaptureSnapshot_Request')
            logger.debug(
                'Failed to import needed modules for type support:\n' +
                traceback.format_exc())
        else:
            cls._CREATE_ROS_MESSAGE = module.create_ros_message_msg__srv__capture_snapshot__request
            cls._CONVERT_FROM_PY = module.convert_from_py_msg__srv__capture_snapshot__request
            cls._CONVERT_TO_PY = module.convert_to_py_msg__srv__capture_snapshot__request
            cls._TYPE_SUPPORT = module.type_support_msg__srv__capture_snapshot__request
            cls._DESTROY_ROS_MESSAGE = module.destroy_ros_message_msg__srv__capture_snapshot__request

    @classmethod
    def __prepare__(cls, name, bases, **kwargs):
        # list constant names here so that they appear in the help text of
        # the message class under "Data and other attributes defined here:"
        # as well as populate each message instance
        return {
        }


class CaptureSnapshot_Request(metaclass=Metaclass_CaptureSnapshot_Request):
    """Message class 'CaptureSnapshot_Request'."""

    __slots__ = [
        '_plant_id',
        '_view_id',
    ]

    _fields_and_field_types = {
        'plant_id': 'int32',
        'view_id': 'int32',
    }

    SLOT_TYPES = (
        rosidl_parser.definition.BasicType('int32'),  # noqa: E501
        rosidl_parser.definition.BasicType('int32'),  # noqa: E501
    )

    def __init__(self, **kwargs):
        assert all('_' + key in self.__slots__ for key in kwargs.keys()), \
            'Invalid arguments passed to constructor: %s' % \
            ', '.join(sorted(k for k in kwargs.keys() if '_' + k not in self.__slots__))
        self.plant_id = kwargs.get('plant_id', int())
        self.view_id = kwargs.get('view_id', int())

    def __repr__(self):
        typename = self.__class__.__module__.split('.')
        typename.pop()
        typename.append(self.__class__.__name__)
        args = []
        for s, t in zip(self.__slots__, self.SLOT_TYPES):
            field = getattr(self, s)
            fieldstr = repr(field)
            # We use Python array type for fields that can be directly stored
            # in them, and "normal" sequences for everything else.  If it is
            # a type that we store in an array, strip off the 'array' portion.
            if (
                isinstance(t, rosidl_parser.definition.AbstractSequence) and
                isinstance(t.value_type, rosidl_parser.definition.BasicType) and
                t.value_type.typename in ['float', 'double', 'int8', 'uint8', 'int16', 'uint16', 'int32', 'uint32', 'int64', 'uint64']
            ):
                if len(field) == 0:
                    fieldstr = '[]'
                else:
                    assert fieldstr.startswith('array(')
                    prefix = "array('X', "
                    suffix = ')'
                    fieldstr = fieldstr[len(prefix):-len(suffix)]
            args.append(s[1:] + '=' + fieldstr)
        return '%s(%s)' % ('.'.join(typename), ', '.join(args))

    def __eq__(self, other):
        if not isinstance(other, self.__class__):
            return False
        if self.plant_id != other.plant_id:
            return False
        if self.view_id != other.view_id:
            return False
        return True

    @classmethod
    def get_fields_and_field_types(cls):
        from copy import copy
        return copy(cls._fields_and_field_types)

    @builtins.property
    def plant_id(self):
        """Message field 'plant_id'."""
        return self._plant_id

    @plant_id.setter
    def plant_id(self, value):
        if __debug__:
            assert \
                isinstance(value, int), \
                "The 'plant_id' field must be of type 'int'"
            assert value >= -2147483648 and value < 2147483648, \
                "The 'plant_id' field must be an integer in [-2147483648, 2147483647]"
        self._plant_id = value

    @builtins.property
    def view_id(self):
        """Message field 'view_id'."""
        return self._view_id

    @view_id.setter
    def view_id(self, value):
        if __debug__:
            assert \
                isinstance(value, int), \
                "The 'view_id' field must be of type 'int'"
            assert value >= -2147483648 and value < 2147483648, \
                "The 'view_id' field must be an integer in [-2147483648, 2147483647]"
        self._view_id = value


# Import statements for member types

# already imported above
# import builtins

# already imported above
# import rosidl_parser.definition


class Metaclass_CaptureSnapshot_Response(type):
    """Metaclass of message 'CaptureSnapshot_Response'."""

    _CREATE_ROS_MESSAGE = None
    _CONVERT_FROM_PY = None
    _CONVERT_TO_PY = None
    _DESTROY_ROS_MESSAGE = None
    _TYPE_SUPPORT = None

    __constants = {
    }

    @classmethod
    def __import_type_support__(cls):
        try:
            from rosidl_generator_py import import_type_support
            module = import_type_support('strawberry_msgs')
        except ImportError:
            import logging
            import traceback
            logger = logging.getLogger(
                'strawberry_msgs.srv.CaptureSnapshot_Response')
            logger.debug(
                'Failed to import needed modules for type support:\n' +
                traceback.format_exc())
        else:
            cls._CREATE_ROS_MESSAGE = module.create_ros_message_msg__srv__capture_snapshot__response
            cls._CONVERT_FROM_PY = module.convert_from_py_msg__srv__capture_snapshot__response
            cls._CONVERT_TO_PY = module.convert_to_py_msg__srv__capture_snapshot__response
            cls._TYPE_SUPPORT = module.type_support_msg__srv__capture_snapshot__response
            cls._DESTROY_ROS_MESSAGE = module.destroy_ros_message_msg__srv__capture_snapshot__response

    @classmethod
    def __prepare__(cls, name, bases, **kwargs):
        # list constant names here so that they appear in the help text of
        # the message class under "Data and other attributes defined here:"
        # as well as populate each message instance
        return {
        }


class CaptureSnapshot_Response(metaclass=Metaclass_CaptureSnapshot_Response):
    """Message class 'CaptureSnapshot_Response'."""

    __slots__ = [
        '_success',
        '_message',
        '_rgb_path',
        '_depth_path',
    ]

    _fields_and_field_types = {
        'success': 'boolean',
        'message': 'string',
        'rgb_path': 'string',
        'depth_path': 'string',
    }

    SLOT_TYPES = (
        rosidl_parser.definition.BasicType('boolean'),  # noqa: E501
        rosidl_parser.definition.UnboundedString(),  # noqa: E501
        rosidl_parser.definition.UnboundedString(),  # noqa: E501
        rosidl_parser.definition.UnboundedString(),  # noqa: E501
    )

    def __init__(self, **kwargs):
        assert all('_' + key in self.__slots__ for key in kwargs.keys()), \
            'Invalid arguments passed to constructor: %s' % \
            ', '.join(sorted(k for k in kwargs.keys() if '_' + k not in self.__slots__))
        self.success = kwargs.get('success', bool())
        self.message = kwargs.get('message', str())
        self.rgb_path = kwargs.get('rgb_path', str())
        self.depth_path = kwargs.get('depth_path', str())

    def __repr__(self):
        typename = self.__class__.__module__.split('.')
        typename.pop()
        typename.append(self.__class__.__name__)
        args = []
        for s, t in zip(self.__slots__, self.SLOT_TYPES):
            field = getattr(self, s)
            fieldstr = repr(field)
            # We use Python array type for fields that can be directly stored
            # in them, and "normal" sequences for everything else.  If it is
            # a type that we store in an array, strip off the 'array' portion.
            if (
                isinstance(t, rosidl_parser.definition.AbstractSequence) and
                isinstance(t.value_type, rosidl_parser.definition.BasicType) and
                t.value_type.typename in ['float', 'double', 'int8', 'uint8', 'int16', 'uint16', 'int32', 'uint32', 'int64', 'uint64']
            ):
                if len(field) == 0:
                    fieldstr = '[]'
                else:
                    assert fieldstr.startswith('array(')
                    prefix = "array('X', "
                    suffix = ')'
                    fieldstr = fieldstr[len(prefix):-len(suffix)]
            args.append(s[1:] + '=' + fieldstr)
        return '%s(%s)' % ('.'.join(typename), ', '.join(args))

    def __eq__(self, other):
        if not isinstance(other, self.__class__):
            return False
        if self.success != other.success:
            return False
        if self.message != other.message:
            return False
        if self.rgb_path != other.rgb_path:
            return False
        if self.depth_path != other.depth_path:
            return False
        return True

    @classmethod
    def get_fields_and_field_types(cls):
        from copy import copy
        return copy(cls._fields_and_field_types)

    @builtins.property
    def success(self):
        """Message field 'success'."""
        return self._success

    @success.setter
    def success(self, value):
        if __debug__:
            assert \
                isinstance(value, bool), \
                "The 'success' field must be of type 'bool'"
        self._success = value

    @builtins.property
    def message(self):
        """Message field 'message'."""
        return self._message

    @message.setter
    def message(self, value):
        if __debug__:
            assert \
                isinstance(value, str), \
                "The 'message' field must be of type 'str'"
        self._message = value

    @builtins.property
    def rgb_path(self):
        """Message field 'rgb_path'."""
        return self._rgb_path

    @rgb_path.setter
    def rgb_path(self, value):
        if __debug__:
            assert \
                isinstance(value, str), \
                "The 'rgb_path' field must be of type 'str'"
        self._rgb_path = value

    @builtins.property
    def depth_path(self):
        """Message field 'depth_path'."""
        return self._depth_path

    @depth_path.setter
    def depth_path(self, value):
        if __debug__:
            assert \
                isinstance(value, str), \
                "The 'depth_path' field must be of type 'str'"
        self._depth_path = value


class Metaclass_CaptureSnapshot(type):
    """Metaclass of service 'CaptureSnapshot'."""

    _TYPE_SUPPORT = None

    @classmethod
    def __import_type_support__(cls):
        try:
            from rosidl_generator_py import import_type_support
            module = import_type_support('strawberry_msgs')
        except ImportError:
            import logging
            import traceback
            logger = logging.getLogger(
                'strawberry_msgs.srv.CaptureSnapshot')
            logger.debug(
                'Failed to import needed modules for type support:\n' +
                traceback.format_exc())
        else:
            cls._TYPE_SUPPORT = module.type_support_srv__srv__capture_snapshot

            from strawberry_msgs.srv import _capture_snapshot
            if _capture_snapshot.Metaclass_CaptureSnapshot_Request._TYPE_SUPPORT is None:
                _capture_snapshot.Metaclass_CaptureSnapshot_Request.__import_type_support__()
            if _capture_snapshot.Metaclass_CaptureSnapshot_Response._TYPE_SUPPORT is None:
                _capture_snapshot.Metaclass_CaptureSnapshot_Response.__import_type_support__()


class CaptureSnapshot(metaclass=Metaclass_CaptureSnapshot):
    from strawberry_msgs.srv._capture_snapshot import CaptureSnapshot_Request as Request
    from strawberry_msgs.srv._capture_snapshot import CaptureSnapshot_Response as Response

    def __init__(self):
        raise NotImplementedError('Service classes can not be instantiated')
