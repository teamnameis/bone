# -*- coding: utf-8 -*-
# Generated by the protocol buffer compiler.  DO NOT EDIT!
# source: bone.proto

import sys
_b=sys.version_info[0]<3 and (lambda x:x) or (lambda x:x.encode('latin1'))
from google.protobuf import descriptor as _descriptor
from google.protobuf import message as _message
from google.protobuf import reflection as _reflection
from google.protobuf import symbol_database as _symbol_database
# @@protoc_insertion_point(imports)

_sym_db = _symbol_database.Default()




DESCRIPTOR = _descriptor.FileDescriptor(
  name='bone.proto',
  package='be',
  syntax='proto3',
  serialized_options=_b('\n\023org.teamnameis.boneB\005MorphP\001Z\004bone'),
  serialized_pb=_b('\n\nbone.proto\x12\x02\x62\x65\"!\n\x05\x46lame\x12\n\n\x02id\x18\x01 \x01(\x05\x12\x0c\n\x04\x64\x61ta\x18\x02 \x01(\x0c\"\x15\n\x05Image\x12\x0c\n\x04\x64\x61ta\x18\x01 \x01(\x0c\x32%\n\x02ML\x12\x1f\n\x05Morph\x12\t.be.Flame\x1a\t.be.Image\"\x00\x42$\n\x13org.teamnameis.boneB\x05MorphP\x01Z\x04\x62oneb\x06proto3')
)




_FLAME = _descriptor.Descriptor(
  name='Flame',
  full_name='be.Flame',
  filename=None,
  file=DESCRIPTOR,
  containing_type=None,
  fields=[
    _descriptor.FieldDescriptor(
      name='id', full_name='be.Flame.id', index=0,
      number=1, type=5, cpp_type=1, label=1,
      has_default_value=False, default_value=0,
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR),
    _descriptor.FieldDescriptor(
      name='data', full_name='be.Flame.data', index=1,
      number=2, type=12, cpp_type=9, label=1,
      has_default_value=False, default_value=_b(""),
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR),
  ],
  extensions=[
  ],
  nested_types=[],
  enum_types=[
  ],
  serialized_options=None,
  is_extendable=False,
  syntax='proto3',
  extension_ranges=[],
  oneofs=[
  ],
  serialized_start=18,
  serialized_end=51,
)


_IMAGE = _descriptor.Descriptor(
  name='Image',
  full_name='be.Image',
  filename=None,
  file=DESCRIPTOR,
  containing_type=None,
  fields=[
    _descriptor.FieldDescriptor(
      name='data', full_name='be.Image.data', index=0,
      number=1, type=12, cpp_type=9, label=1,
      has_default_value=False, default_value=_b(""),
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR),
  ],
  extensions=[
  ],
  nested_types=[],
  enum_types=[
  ],
  serialized_options=None,
  is_extendable=False,
  syntax='proto3',
  extension_ranges=[],
  oneofs=[
  ],
  serialized_start=53,
  serialized_end=74,
)

DESCRIPTOR.message_types_by_name['Flame'] = _FLAME
DESCRIPTOR.message_types_by_name['Image'] = _IMAGE
_sym_db.RegisterFileDescriptor(DESCRIPTOR)

Flame = _reflection.GeneratedProtocolMessageType('Flame', (_message.Message,), {
  'DESCRIPTOR' : _FLAME,
  '__module__' : 'bone_pb2'
  # @@protoc_insertion_point(class_scope:be.Flame)
  })
_sym_db.RegisterMessage(Flame)

Image = _reflection.GeneratedProtocolMessageType('Image', (_message.Message,), {
  'DESCRIPTOR' : _IMAGE,
  '__module__' : 'bone_pb2'
  # @@protoc_insertion_point(class_scope:be.Image)
  })
_sym_db.RegisterMessage(Image)


DESCRIPTOR._options = None

_ML = _descriptor.ServiceDescriptor(
  name='ML',
  full_name='be.ML',
  file=DESCRIPTOR,
  index=0,
  serialized_options=None,
  serialized_start=76,
  serialized_end=113,
  methods=[
  _descriptor.MethodDescriptor(
    name='Morph',
    full_name='be.ML.Morph',
    index=0,
    containing_service=None,
    input_type=_FLAME,
    output_type=_IMAGE,
    serialized_options=None,
  ),
])
_sym_db.RegisterServiceDescriptor(_ML)

DESCRIPTOR.services_by_name['ML'] = _ML

# @@protoc_insertion_point(module_scope)
