// Generated by the protocol buffer compiler.  DO NOT EDIT!
// source: brpc/rpc_dump.proto

#ifndef GOOGLE_PROTOBUF_INCLUDED_brpc_2frpc_5fdump_2eproto
#define GOOGLE_PROTOBUF_INCLUDED_brpc_2frpc_5fdump_2eproto

#include <limits>
#include <string>

#include <google/protobuf/port_def.inc>
#if PROTOBUF_VERSION < 3021000
#error This file was generated by a newer version of protoc which is
#error incompatible with your Protocol Buffer headers. Please update
#error your headers.
#endif
#if 3021009 < PROTOBUF_MIN_PROTOC_VERSION
#error This file was generated by an older version of protoc which is
#error incompatible with your Protocol Buffer headers. Please
#error regenerate this file with a newer version of protoc.
#endif

#include <google/protobuf/port_undef.inc>
#include <google/protobuf/io/coded_stream.h>
#include <google/protobuf/arena.h>
#include <google/protobuf/arenastring.h>
#include <google/protobuf/generated_message_util.h>
#include <google/protobuf/metadata_lite.h>
#include <google/protobuf/generated_message_reflection.h>
#include <google/protobuf/message.h>
#include <google/protobuf/repeated_field.h>  // IWYU pragma: export
#include <google/protobuf/extension_set.h>  // IWYU pragma: export
#include <google/protobuf/unknown_field_set.h>
#include "brpc/options.pb.h"
// @@protoc_insertion_point(includes)
#include <google/protobuf/port_def.inc>
#define PROTOBUF_INTERNAL_EXPORT_brpc_2frpc_5fdump_2eproto
PROTOBUF_NAMESPACE_OPEN
namespace internal {
class AnyMetadata;
}  // namespace internal
PROTOBUF_NAMESPACE_CLOSE

// Internal implementation detail -- do not use these members.
struct TableStruct_brpc_2frpc_5fdump_2eproto {
  static const uint32_t offsets[];
};
extern const ::PROTOBUF_NAMESPACE_ID::internal::DescriptorTable descriptor_table_brpc_2frpc_5fdump_2eproto;
namespace brpc {
class RpcDumpMeta;
struct RpcDumpMetaDefaultTypeInternal;
extern RpcDumpMetaDefaultTypeInternal _RpcDumpMeta_default_instance_;
}  // namespace brpc
PROTOBUF_NAMESPACE_OPEN
template<> ::brpc::RpcDumpMeta* Arena::CreateMaybeMessage<::brpc::RpcDumpMeta>(Arena*);
PROTOBUF_NAMESPACE_CLOSE
namespace brpc {

// ===================================================================

class RpcDumpMeta final :
    public ::PROTOBUF_NAMESPACE_ID::Message /* @@protoc_insertion_point(class_definition:brpc.RpcDumpMeta) */ {
 public:
  inline RpcDumpMeta() : RpcDumpMeta(nullptr) {}
  ~RpcDumpMeta() override;
  explicit PROTOBUF_CONSTEXPR RpcDumpMeta(::PROTOBUF_NAMESPACE_ID::internal::ConstantInitialized);

  RpcDumpMeta(const RpcDumpMeta& from);
  RpcDumpMeta(RpcDumpMeta&& from) noexcept
    : RpcDumpMeta() {
    *this = ::std::move(from);
  }

  inline RpcDumpMeta& operator=(const RpcDumpMeta& from) {
    CopyFrom(from);
    return *this;
  }
  inline RpcDumpMeta& operator=(RpcDumpMeta&& from) noexcept {
    if (this == &from) return *this;
    if (GetOwningArena() == from.GetOwningArena()
  #ifdef PROTOBUF_FORCE_COPY_IN_MOVE
        && GetOwningArena() != nullptr
  #endif  // !PROTOBUF_FORCE_COPY_IN_MOVE
    ) {
      InternalSwap(&from);
    } else {
      CopyFrom(from);
    }
    return *this;
  }

  inline const ::PROTOBUF_NAMESPACE_ID::UnknownFieldSet& unknown_fields() const {
    return _internal_metadata_.unknown_fields<::PROTOBUF_NAMESPACE_ID::UnknownFieldSet>(::PROTOBUF_NAMESPACE_ID::UnknownFieldSet::default_instance);
  }
  inline ::PROTOBUF_NAMESPACE_ID::UnknownFieldSet* mutable_unknown_fields() {
    return _internal_metadata_.mutable_unknown_fields<::PROTOBUF_NAMESPACE_ID::UnknownFieldSet>();
  }

  static const ::PROTOBUF_NAMESPACE_ID::Descriptor* descriptor() {
    return GetDescriptor();
  }
  static const ::PROTOBUF_NAMESPACE_ID::Descriptor* GetDescriptor() {
    return default_instance().GetMetadata().descriptor;
  }
  static const ::PROTOBUF_NAMESPACE_ID::Reflection* GetReflection() {
    return default_instance().GetMetadata().reflection;
  }
  static const RpcDumpMeta& default_instance() {
    return *internal_default_instance();
  }
  static inline const RpcDumpMeta* internal_default_instance() {
    return reinterpret_cast<const RpcDumpMeta*>(
               &_RpcDumpMeta_default_instance_);
  }
  static constexpr int kIndexInFileMessages =
    0;

  friend void swap(RpcDumpMeta& a, RpcDumpMeta& b) {
    a.Swap(&b);
  }
  inline void Swap(RpcDumpMeta* other) {
    if (other == this) return;
  #ifdef PROTOBUF_FORCE_COPY_IN_SWAP
    if (GetOwningArena() != nullptr &&
        GetOwningArena() == other->GetOwningArena()) {
   #else  // PROTOBUF_FORCE_COPY_IN_SWAP
    if (GetOwningArena() == other->GetOwningArena()) {
  #endif  // !PROTOBUF_FORCE_COPY_IN_SWAP
      InternalSwap(other);
    } else {
      ::PROTOBUF_NAMESPACE_ID::internal::GenericSwap(this, other);
    }
  }
  void UnsafeArenaSwap(RpcDumpMeta* other) {
    if (other == this) return;
    GOOGLE_DCHECK(GetOwningArena() == other->GetOwningArena());
    InternalSwap(other);
  }

  // implements Message ----------------------------------------------

  RpcDumpMeta* New(::PROTOBUF_NAMESPACE_ID::Arena* arena = nullptr) const final {
    return CreateMaybeMessage<RpcDumpMeta>(arena);
  }
  using ::PROTOBUF_NAMESPACE_ID::Message::CopyFrom;
  void CopyFrom(const RpcDumpMeta& from);
  using ::PROTOBUF_NAMESPACE_ID::Message::MergeFrom;
  void MergeFrom( const RpcDumpMeta& from) {
    RpcDumpMeta::MergeImpl(*this, from);
  }
  private:
  static void MergeImpl(::PROTOBUF_NAMESPACE_ID::Message& to_msg, const ::PROTOBUF_NAMESPACE_ID::Message& from_msg);
  public:
  PROTOBUF_ATTRIBUTE_REINITIALIZES void Clear() final;
  bool IsInitialized() const final;

  size_t ByteSizeLong() const final;
  const char* _InternalParse(const char* ptr, ::PROTOBUF_NAMESPACE_ID::internal::ParseContext* ctx) final;
  uint8_t* _InternalSerialize(
      uint8_t* target, ::PROTOBUF_NAMESPACE_ID::io::EpsCopyOutputStream* stream) const final;
  int GetCachedSize() const final { return _impl_._cached_size_.Get(); }

  private:
  void SharedCtor(::PROTOBUF_NAMESPACE_ID::Arena* arena, bool is_message_owned);
  void SharedDtor();
  void SetCachedSize(int size) const final;
  void InternalSwap(RpcDumpMeta* other);

  private:
  friend class ::PROTOBUF_NAMESPACE_ID::internal::AnyMetadata;
  static ::PROTOBUF_NAMESPACE_ID::StringPiece FullMessageName() {
    return "brpc.RpcDumpMeta";
  }
  protected:
  explicit RpcDumpMeta(::PROTOBUF_NAMESPACE_ID::Arena* arena,
                       bool is_message_owned = false);
  public:

  static const ClassData _class_data_;
  const ::PROTOBUF_NAMESPACE_ID::Message::ClassData*GetClassData() const final;

  ::PROTOBUF_NAMESPACE_ID::Metadata GetMetadata() const final;

  // nested types ----------------------------------------------------

  // accessors -------------------------------------------------------

  enum : int {
    kServiceNameFieldNumber = 1,
    kMethodNameFieldNumber = 2,
    kAuthenticationDataFieldNumber = 7,
    kUserDataFieldNumber = 8,
    kNsheadFieldNumber = 9,
    kMethodIndexFieldNumber = 3,
    kCompressTypeFieldNumber = 4,
    kProtocolTypeFieldNumber = 5,
    kAttachmentSizeFieldNumber = 6,
  };
  // optional string service_name = 1;
  bool has_service_name() const;
  private:
  bool _internal_has_service_name() const;
  public:
  void clear_service_name();
  const std::string& service_name() const;
  template <typename ArgT0 = const std::string&, typename... ArgT>
  void set_service_name(ArgT0&& arg0, ArgT... args);
  std::string* mutable_service_name();
  PROTOBUF_NODISCARD std::string* release_service_name();
  void set_allocated_service_name(std::string* service_name);
  private:
  const std::string& _internal_service_name() const;
  inline PROTOBUF_ALWAYS_INLINE void _internal_set_service_name(const std::string& value);
  std::string* _internal_mutable_service_name();
  public:

  // optional string method_name = 2;
  bool has_method_name() const;
  private:
  bool _internal_has_method_name() const;
  public:
  void clear_method_name();
  const std::string& method_name() const;
  template <typename ArgT0 = const std::string&, typename... ArgT>
  void set_method_name(ArgT0&& arg0, ArgT... args);
  std::string* mutable_method_name();
  PROTOBUF_NODISCARD std::string* release_method_name();
  void set_allocated_method_name(std::string* method_name);
  private:
  const std::string& _internal_method_name() const;
  inline PROTOBUF_ALWAYS_INLINE void _internal_set_method_name(const std::string& value);
  std::string* _internal_mutable_method_name();
  public:

  // optional bytes authentication_data = 7;
  bool has_authentication_data() const;
  private:
  bool _internal_has_authentication_data() const;
  public:
  void clear_authentication_data();
  const std::string& authentication_data() const;
  template <typename ArgT0 = const std::string&, typename... ArgT>
  void set_authentication_data(ArgT0&& arg0, ArgT... args);
  std::string* mutable_authentication_data();
  PROTOBUF_NODISCARD std::string* release_authentication_data();
  void set_allocated_authentication_data(std::string* authentication_data);
  private:
  const std::string& _internal_authentication_data() const;
  inline PROTOBUF_ALWAYS_INLINE void _internal_set_authentication_data(const std::string& value);
  std::string* _internal_mutable_authentication_data();
  public:

  // optional bytes user_data = 8;
  bool has_user_data() const;
  private:
  bool _internal_has_user_data() const;
  public:
  void clear_user_data();
  const std::string& user_data() const;
  template <typename ArgT0 = const std::string&, typename... ArgT>
  void set_user_data(ArgT0&& arg0, ArgT... args);
  std::string* mutable_user_data();
  PROTOBUF_NODISCARD std::string* release_user_data();
  void set_allocated_user_data(std::string* user_data);
  private:
  const std::string& _internal_user_data() const;
  inline PROTOBUF_ALWAYS_INLINE void _internal_set_user_data(const std::string& value);
  std::string* _internal_mutable_user_data();
  public:

  // optional bytes nshead = 9;
  bool has_nshead() const;
  private:
  bool _internal_has_nshead() const;
  public:
  void clear_nshead();
  const std::string& nshead() const;
  template <typename ArgT0 = const std::string&, typename... ArgT>
  void set_nshead(ArgT0&& arg0, ArgT... args);
  std::string* mutable_nshead();
  PROTOBUF_NODISCARD std::string* release_nshead();
  void set_allocated_nshead(std::string* nshead);
  private:
  const std::string& _internal_nshead() const;
  inline PROTOBUF_ALWAYS_INLINE void _internal_set_nshead(const std::string& value);
  std::string* _internal_mutable_nshead();
  public:

  // optional int32 method_index = 3;
  bool has_method_index() const;
  private:
  bool _internal_has_method_index() const;
  public:
  void clear_method_index();
  int32_t method_index() const;
  void set_method_index(int32_t value);
  private:
  int32_t _internal_method_index() const;
  void _internal_set_method_index(int32_t value);
  public:

  // optional .brpc.CompressType compress_type = 4;
  bool has_compress_type() const;
  private:
  bool _internal_has_compress_type() const;
  public:
  void clear_compress_type();
  ::brpc::CompressType compress_type() const;
  void set_compress_type(::brpc::CompressType value);
  private:
  ::brpc::CompressType _internal_compress_type() const;
  void _internal_set_compress_type(::brpc::CompressType value);
  public:

  // optional .brpc.ProtocolType protocol_type = 5;
  bool has_protocol_type() const;
  private:
  bool _internal_has_protocol_type() const;
  public:
  void clear_protocol_type();
  ::brpc::ProtocolType protocol_type() const;
  void set_protocol_type(::brpc::ProtocolType value);
  private:
  ::brpc::ProtocolType _internal_protocol_type() const;
  void _internal_set_protocol_type(::brpc::ProtocolType value);
  public:

  // optional int32 attachment_size = 6;
  bool has_attachment_size() const;
  private:
  bool _internal_has_attachment_size() const;
  public:
  void clear_attachment_size();
  int32_t attachment_size() const;
  void set_attachment_size(int32_t value);
  private:
  int32_t _internal_attachment_size() const;
  void _internal_set_attachment_size(int32_t value);
  public:

  // @@protoc_insertion_point(class_scope:brpc.RpcDumpMeta)
 private:
  class _Internal;

  template <typename T> friend class ::PROTOBUF_NAMESPACE_ID::Arena::InternalHelper;
  typedef void InternalArenaConstructable_;
  typedef void DestructorSkippable_;
  struct Impl_ {
    ::PROTOBUF_NAMESPACE_ID::internal::HasBits<1> _has_bits_;
    mutable ::PROTOBUF_NAMESPACE_ID::internal::CachedSize _cached_size_;
    ::PROTOBUF_NAMESPACE_ID::internal::ArenaStringPtr service_name_;
    ::PROTOBUF_NAMESPACE_ID::internal::ArenaStringPtr method_name_;
    ::PROTOBUF_NAMESPACE_ID::internal::ArenaStringPtr authentication_data_;
    ::PROTOBUF_NAMESPACE_ID::internal::ArenaStringPtr user_data_;
    ::PROTOBUF_NAMESPACE_ID::internal::ArenaStringPtr nshead_;
    int32_t method_index_;
    int compress_type_;
    int protocol_type_;
    int32_t attachment_size_;
  };
  union { Impl_ _impl_; };
  friend struct ::TableStruct_brpc_2frpc_5fdump_2eproto;
};
// ===================================================================


// ===================================================================

#ifdef __GNUC__
  #pragma GCC diagnostic push
  #pragma GCC diagnostic ignored "-Wstrict-aliasing"
#endif  // __GNUC__
// RpcDumpMeta

// optional string service_name = 1;
inline bool RpcDumpMeta::_internal_has_service_name() const {
  bool value = (_impl_._has_bits_[0] & 0x00000001u) != 0;
  return value;
}
inline bool RpcDumpMeta::has_service_name() const {
  return _internal_has_service_name();
}
inline void RpcDumpMeta::clear_service_name() {
  _impl_.service_name_.ClearToEmpty();
  _impl_._has_bits_[0] &= ~0x00000001u;
}
inline const std::string& RpcDumpMeta::service_name() const {
  // @@protoc_insertion_point(field_get:brpc.RpcDumpMeta.service_name)
  return _internal_service_name();
}
template <typename ArgT0, typename... ArgT>
inline PROTOBUF_ALWAYS_INLINE
void RpcDumpMeta::set_service_name(ArgT0&& arg0, ArgT... args) {
 _impl_._has_bits_[0] |= 0x00000001u;
 _impl_.service_name_.Set(static_cast<ArgT0 &&>(arg0), args..., GetArenaForAllocation());
  // @@protoc_insertion_point(field_set:brpc.RpcDumpMeta.service_name)
}
inline std::string* RpcDumpMeta::mutable_service_name() {
  std::string* _s = _internal_mutable_service_name();
  // @@protoc_insertion_point(field_mutable:brpc.RpcDumpMeta.service_name)
  return _s;
}
inline const std::string& RpcDumpMeta::_internal_service_name() const {
  return _impl_.service_name_.Get();
}
inline void RpcDumpMeta::_internal_set_service_name(const std::string& value) {
  _impl_._has_bits_[0] |= 0x00000001u;
  _impl_.service_name_.Set(value, GetArenaForAllocation());
}
inline std::string* RpcDumpMeta::_internal_mutable_service_name() {
  _impl_._has_bits_[0] |= 0x00000001u;
  return _impl_.service_name_.Mutable(GetArenaForAllocation());
}
inline std::string* RpcDumpMeta::release_service_name() {
  // @@protoc_insertion_point(field_release:brpc.RpcDumpMeta.service_name)
  if (!_internal_has_service_name()) {
    return nullptr;
  }
  _impl_._has_bits_[0] &= ~0x00000001u;
  auto* p = _impl_.service_name_.Release();
#ifdef PROTOBUF_FORCE_COPY_DEFAULT_STRING
  if (_impl_.service_name_.IsDefault()) {
    _impl_.service_name_.Set("", GetArenaForAllocation());
  }
#endif // PROTOBUF_FORCE_COPY_DEFAULT_STRING
  return p;
}
inline void RpcDumpMeta::set_allocated_service_name(std::string* service_name) {
  if (service_name != nullptr) {
    _impl_._has_bits_[0] |= 0x00000001u;
  } else {
    _impl_._has_bits_[0] &= ~0x00000001u;
  }
  _impl_.service_name_.SetAllocated(service_name, GetArenaForAllocation());
#ifdef PROTOBUF_FORCE_COPY_DEFAULT_STRING
  if (_impl_.service_name_.IsDefault()) {
    _impl_.service_name_.Set("", GetArenaForAllocation());
  }
#endif // PROTOBUF_FORCE_COPY_DEFAULT_STRING
  // @@protoc_insertion_point(field_set_allocated:brpc.RpcDumpMeta.service_name)
}

// optional string method_name = 2;
inline bool RpcDumpMeta::_internal_has_method_name() const {
  bool value = (_impl_._has_bits_[0] & 0x00000002u) != 0;
  return value;
}
inline bool RpcDumpMeta::has_method_name() const {
  return _internal_has_method_name();
}
inline void RpcDumpMeta::clear_method_name() {
  _impl_.method_name_.ClearToEmpty();
  _impl_._has_bits_[0] &= ~0x00000002u;
}
inline const std::string& RpcDumpMeta::method_name() const {
  // @@protoc_insertion_point(field_get:brpc.RpcDumpMeta.method_name)
  return _internal_method_name();
}
template <typename ArgT0, typename... ArgT>
inline PROTOBUF_ALWAYS_INLINE
void RpcDumpMeta::set_method_name(ArgT0&& arg0, ArgT... args) {
 _impl_._has_bits_[0] |= 0x00000002u;
 _impl_.method_name_.Set(static_cast<ArgT0 &&>(arg0), args..., GetArenaForAllocation());
  // @@protoc_insertion_point(field_set:brpc.RpcDumpMeta.method_name)
}
inline std::string* RpcDumpMeta::mutable_method_name() {
  std::string* _s = _internal_mutable_method_name();
  // @@protoc_insertion_point(field_mutable:brpc.RpcDumpMeta.method_name)
  return _s;
}
inline const std::string& RpcDumpMeta::_internal_method_name() const {
  return _impl_.method_name_.Get();
}
inline void RpcDumpMeta::_internal_set_method_name(const std::string& value) {
  _impl_._has_bits_[0] |= 0x00000002u;
  _impl_.method_name_.Set(value, GetArenaForAllocation());
}
inline std::string* RpcDumpMeta::_internal_mutable_method_name() {
  _impl_._has_bits_[0] |= 0x00000002u;
  return _impl_.method_name_.Mutable(GetArenaForAllocation());
}
inline std::string* RpcDumpMeta::release_method_name() {
  // @@protoc_insertion_point(field_release:brpc.RpcDumpMeta.method_name)
  if (!_internal_has_method_name()) {
    return nullptr;
  }
  _impl_._has_bits_[0] &= ~0x00000002u;
  auto* p = _impl_.method_name_.Release();
#ifdef PROTOBUF_FORCE_COPY_DEFAULT_STRING
  if (_impl_.method_name_.IsDefault()) {
    _impl_.method_name_.Set("", GetArenaForAllocation());
  }
#endif // PROTOBUF_FORCE_COPY_DEFAULT_STRING
  return p;
}
inline void RpcDumpMeta::set_allocated_method_name(std::string* method_name) {
  if (method_name != nullptr) {
    _impl_._has_bits_[0] |= 0x00000002u;
  } else {
    _impl_._has_bits_[0] &= ~0x00000002u;
  }
  _impl_.method_name_.SetAllocated(method_name, GetArenaForAllocation());
#ifdef PROTOBUF_FORCE_COPY_DEFAULT_STRING
  if (_impl_.method_name_.IsDefault()) {
    _impl_.method_name_.Set("", GetArenaForAllocation());
  }
#endif // PROTOBUF_FORCE_COPY_DEFAULT_STRING
  // @@protoc_insertion_point(field_set_allocated:brpc.RpcDumpMeta.method_name)
}

// optional int32 method_index = 3;
inline bool RpcDumpMeta::_internal_has_method_index() const {
  bool value = (_impl_._has_bits_[0] & 0x00000020u) != 0;
  return value;
}
inline bool RpcDumpMeta::has_method_index() const {
  return _internal_has_method_index();
}
inline void RpcDumpMeta::clear_method_index() {
  _impl_.method_index_ = 0;
  _impl_._has_bits_[0] &= ~0x00000020u;
}
inline int32_t RpcDumpMeta::_internal_method_index() const {
  return _impl_.method_index_;
}
inline int32_t RpcDumpMeta::method_index() const {
  // @@protoc_insertion_point(field_get:brpc.RpcDumpMeta.method_index)
  return _internal_method_index();
}
inline void RpcDumpMeta::_internal_set_method_index(int32_t value) {
  _impl_._has_bits_[0] |= 0x00000020u;
  _impl_.method_index_ = value;
}
inline void RpcDumpMeta::set_method_index(int32_t value) {
  _internal_set_method_index(value);
  // @@protoc_insertion_point(field_set:brpc.RpcDumpMeta.method_index)
}

// optional .brpc.CompressType compress_type = 4;
inline bool RpcDumpMeta::_internal_has_compress_type() const {
  bool value = (_impl_._has_bits_[0] & 0x00000040u) != 0;
  return value;
}
inline bool RpcDumpMeta::has_compress_type() const {
  return _internal_has_compress_type();
}
inline void RpcDumpMeta::clear_compress_type() {
  _impl_.compress_type_ = 0;
  _impl_._has_bits_[0] &= ~0x00000040u;
}
inline ::brpc::CompressType RpcDumpMeta::_internal_compress_type() const {
  return static_cast< ::brpc::CompressType >(_impl_.compress_type_);
}
inline ::brpc::CompressType RpcDumpMeta::compress_type() const {
  // @@protoc_insertion_point(field_get:brpc.RpcDumpMeta.compress_type)
  return _internal_compress_type();
}
inline void RpcDumpMeta::_internal_set_compress_type(::brpc::CompressType value) {
  assert(::brpc::CompressType_IsValid(value));
  _impl_._has_bits_[0] |= 0x00000040u;
  _impl_.compress_type_ = value;
}
inline void RpcDumpMeta::set_compress_type(::brpc::CompressType value) {
  _internal_set_compress_type(value);
  // @@protoc_insertion_point(field_set:brpc.RpcDumpMeta.compress_type)
}

// optional .brpc.ProtocolType protocol_type = 5;
inline bool RpcDumpMeta::_internal_has_protocol_type() const {
  bool value = (_impl_._has_bits_[0] & 0x00000080u) != 0;
  return value;
}
inline bool RpcDumpMeta::has_protocol_type() const {
  return _internal_has_protocol_type();
}
inline void RpcDumpMeta::clear_protocol_type() {
  _impl_.protocol_type_ = 0;
  _impl_._has_bits_[0] &= ~0x00000080u;
}
inline ::brpc::ProtocolType RpcDumpMeta::_internal_protocol_type() const {
  return static_cast< ::brpc::ProtocolType >(_impl_.protocol_type_);
}
inline ::brpc::ProtocolType RpcDumpMeta::protocol_type() const {
  // @@protoc_insertion_point(field_get:brpc.RpcDumpMeta.protocol_type)
  return _internal_protocol_type();
}
inline void RpcDumpMeta::_internal_set_protocol_type(::brpc::ProtocolType value) {
  assert(::brpc::ProtocolType_IsValid(value));
  _impl_._has_bits_[0] |= 0x00000080u;
  _impl_.protocol_type_ = value;
}
inline void RpcDumpMeta::set_protocol_type(::brpc::ProtocolType value) {
  _internal_set_protocol_type(value);
  // @@protoc_insertion_point(field_set:brpc.RpcDumpMeta.protocol_type)
}

// optional int32 attachment_size = 6;
inline bool RpcDumpMeta::_internal_has_attachment_size() const {
  bool value = (_impl_._has_bits_[0] & 0x00000100u) != 0;
  return value;
}
inline bool RpcDumpMeta::has_attachment_size() const {
  return _internal_has_attachment_size();
}
inline void RpcDumpMeta::clear_attachment_size() {
  _impl_.attachment_size_ = 0;
  _impl_._has_bits_[0] &= ~0x00000100u;
}
inline int32_t RpcDumpMeta::_internal_attachment_size() const {
  return _impl_.attachment_size_;
}
inline int32_t RpcDumpMeta::attachment_size() const {
  // @@protoc_insertion_point(field_get:brpc.RpcDumpMeta.attachment_size)
  return _internal_attachment_size();
}
inline void RpcDumpMeta::_internal_set_attachment_size(int32_t value) {
  _impl_._has_bits_[0] |= 0x00000100u;
  _impl_.attachment_size_ = value;
}
inline void RpcDumpMeta::set_attachment_size(int32_t value) {
  _internal_set_attachment_size(value);
  // @@protoc_insertion_point(field_set:brpc.RpcDumpMeta.attachment_size)
}

// optional bytes authentication_data = 7;
inline bool RpcDumpMeta::_internal_has_authentication_data() const {
  bool value = (_impl_._has_bits_[0] & 0x00000004u) != 0;
  return value;
}
inline bool RpcDumpMeta::has_authentication_data() const {
  return _internal_has_authentication_data();
}
inline void RpcDumpMeta::clear_authentication_data() {
  _impl_.authentication_data_.ClearToEmpty();
  _impl_._has_bits_[0] &= ~0x00000004u;
}
inline const std::string& RpcDumpMeta::authentication_data() const {
  // @@protoc_insertion_point(field_get:brpc.RpcDumpMeta.authentication_data)
  return _internal_authentication_data();
}
template <typename ArgT0, typename... ArgT>
inline PROTOBUF_ALWAYS_INLINE
void RpcDumpMeta::set_authentication_data(ArgT0&& arg0, ArgT... args) {
 _impl_._has_bits_[0] |= 0x00000004u;
 _impl_.authentication_data_.SetBytes(static_cast<ArgT0 &&>(arg0), args..., GetArenaForAllocation());
  // @@protoc_insertion_point(field_set:brpc.RpcDumpMeta.authentication_data)
}
inline std::string* RpcDumpMeta::mutable_authentication_data() {
  std::string* _s = _internal_mutable_authentication_data();
  // @@protoc_insertion_point(field_mutable:brpc.RpcDumpMeta.authentication_data)
  return _s;
}
inline const std::string& RpcDumpMeta::_internal_authentication_data() const {
  return _impl_.authentication_data_.Get();
}
inline void RpcDumpMeta::_internal_set_authentication_data(const std::string& value) {
  _impl_._has_bits_[0] |= 0x00000004u;
  _impl_.authentication_data_.Set(value, GetArenaForAllocation());
}
inline std::string* RpcDumpMeta::_internal_mutable_authentication_data() {
  _impl_._has_bits_[0] |= 0x00000004u;
  return _impl_.authentication_data_.Mutable(GetArenaForAllocation());
}
inline std::string* RpcDumpMeta::release_authentication_data() {
  // @@protoc_insertion_point(field_release:brpc.RpcDumpMeta.authentication_data)
  if (!_internal_has_authentication_data()) {
    return nullptr;
  }
  _impl_._has_bits_[0] &= ~0x00000004u;
  auto* p = _impl_.authentication_data_.Release();
#ifdef PROTOBUF_FORCE_COPY_DEFAULT_STRING
  if (_impl_.authentication_data_.IsDefault()) {
    _impl_.authentication_data_.Set("", GetArenaForAllocation());
  }
#endif // PROTOBUF_FORCE_COPY_DEFAULT_STRING
  return p;
}
inline void RpcDumpMeta::set_allocated_authentication_data(std::string* authentication_data) {
  if (authentication_data != nullptr) {
    _impl_._has_bits_[0] |= 0x00000004u;
  } else {
    _impl_._has_bits_[0] &= ~0x00000004u;
  }
  _impl_.authentication_data_.SetAllocated(authentication_data, GetArenaForAllocation());
#ifdef PROTOBUF_FORCE_COPY_DEFAULT_STRING
  if (_impl_.authentication_data_.IsDefault()) {
    _impl_.authentication_data_.Set("", GetArenaForAllocation());
  }
#endif // PROTOBUF_FORCE_COPY_DEFAULT_STRING
  // @@protoc_insertion_point(field_set_allocated:brpc.RpcDumpMeta.authentication_data)
}

// optional bytes user_data = 8;
inline bool RpcDumpMeta::_internal_has_user_data() const {
  bool value = (_impl_._has_bits_[0] & 0x00000008u) != 0;
  return value;
}
inline bool RpcDumpMeta::has_user_data() const {
  return _internal_has_user_data();
}
inline void RpcDumpMeta::clear_user_data() {
  _impl_.user_data_.ClearToEmpty();
  _impl_._has_bits_[0] &= ~0x00000008u;
}
inline const std::string& RpcDumpMeta::user_data() const {
  // @@protoc_insertion_point(field_get:brpc.RpcDumpMeta.user_data)
  return _internal_user_data();
}
template <typename ArgT0, typename... ArgT>
inline PROTOBUF_ALWAYS_INLINE
void RpcDumpMeta::set_user_data(ArgT0&& arg0, ArgT... args) {
 _impl_._has_bits_[0] |= 0x00000008u;
 _impl_.user_data_.SetBytes(static_cast<ArgT0 &&>(arg0), args..., GetArenaForAllocation());
  // @@protoc_insertion_point(field_set:brpc.RpcDumpMeta.user_data)
}
inline std::string* RpcDumpMeta::mutable_user_data() {
  std::string* _s = _internal_mutable_user_data();
  // @@protoc_insertion_point(field_mutable:brpc.RpcDumpMeta.user_data)
  return _s;
}
inline const std::string& RpcDumpMeta::_internal_user_data() const {
  return _impl_.user_data_.Get();
}
inline void RpcDumpMeta::_internal_set_user_data(const std::string& value) {
  _impl_._has_bits_[0] |= 0x00000008u;
  _impl_.user_data_.Set(value, GetArenaForAllocation());
}
inline std::string* RpcDumpMeta::_internal_mutable_user_data() {
  _impl_._has_bits_[0] |= 0x00000008u;
  return _impl_.user_data_.Mutable(GetArenaForAllocation());
}
inline std::string* RpcDumpMeta::release_user_data() {
  // @@protoc_insertion_point(field_release:brpc.RpcDumpMeta.user_data)
  if (!_internal_has_user_data()) {
    return nullptr;
  }
  _impl_._has_bits_[0] &= ~0x00000008u;
  auto* p = _impl_.user_data_.Release();
#ifdef PROTOBUF_FORCE_COPY_DEFAULT_STRING
  if (_impl_.user_data_.IsDefault()) {
    _impl_.user_data_.Set("", GetArenaForAllocation());
  }
#endif // PROTOBUF_FORCE_COPY_DEFAULT_STRING
  return p;
}
inline void RpcDumpMeta::set_allocated_user_data(std::string* user_data) {
  if (user_data != nullptr) {
    _impl_._has_bits_[0] |= 0x00000008u;
  } else {
    _impl_._has_bits_[0] &= ~0x00000008u;
  }
  _impl_.user_data_.SetAllocated(user_data, GetArenaForAllocation());
#ifdef PROTOBUF_FORCE_COPY_DEFAULT_STRING
  if (_impl_.user_data_.IsDefault()) {
    _impl_.user_data_.Set("", GetArenaForAllocation());
  }
#endif // PROTOBUF_FORCE_COPY_DEFAULT_STRING
  // @@protoc_insertion_point(field_set_allocated:brpc.RpcDumpMeta.user_data)
}

// optional bytes nshead = 9;
inline bool RpcDumpMeta::_internal_has_nshead() const {
  bool value = (_impl_._has_bits_[0] & 0x00000010u) != 0;
  return value;
}
inline bool RpcDumpMeta::has_nshead() const {
  return _internal_has_nshead();
}
inline void RpcDumpMeta::clear_nshead() {
  _impl_.nshead_.ClearToEmpty();
  _impl_._has_bits_[0] &= ~0x00000010u;
}
inline const std::string& RpcDumpMeta::nshead() const {
  // @@protoc_insertion_point(field_get:brpc.RpcDumpMeta.nshead)
  return _internal_nshead();
}
template <typename ArgT0, typename... ArgT>
inline PROTOBUF_ALWAYS_INLINE
void RpcDumpMeta::set_nshead(ArgT0&& arg0, ArgT... args) {
 _impl_._has_bits_[0] |= 0x00000010u;
 _impl_.nshead_.SetBytes(static_cast<ArgT0 &&>(arg0), args..., GetArenaForAllocation());
  // @@protoc_insertion_point(field_set:brpc.RpcDumpMeta.nshead)
}
inline std::string* RpcDumpMeta::mutable_nshead() {
  std::string* _s = _internal_mutable_nshead();
  // @@protoc_insertion_point(field_mutable:brpc.RpcDumpMeta.nshead)
  return _s;
}
inline const std::string& RpcDumpMeta::_internal_nshead() const {
  return _impl_.nshead_.Get();
}
inline void RpcDumpMeta::_internal_set_nshead(const std::string& value) {
  _impl_._has_bits_[0] |= 0x00000010u;
  _impl_.nshead_.Set(value, GetArenaForAllocation());
}
inline std::string* RpcDumpMeta::_internal_mutable_nshead() {
  _impl_._has_bits_[0] |= 0x00000010u;
  return _impl_.nshead_.Mutable(GetArenaForAllocation());
}
inline std::string* RpcDumpMeta::release_nshead() {
  // @@protoc_insertion_point(field_release:brpc.RpcDumpMeta.nshead)
  if (!_internal_has_nshead()) {
    return nullptr;
  }
  _impl_._has_bits_[0] &= ~0x00000010u;
  auto* p = _impl_.nshead_.Release();
#ifdef PROTOBUF_FORCE_COPY_DEFAULT_STRING
  if (_impl_.nshead_.IsDefault()) {
    _impl_.nshead_.Set("", GetArenaForAllocation());
  }
#endif // PROTOBUF_FORCE_COPY_DEFAULT_STRING
  return p;
}
inline void RpcDumpMeta::set_allocated_nshead(std::string* nshead) {
  if (nshead != nullptr) {
    _impl_._has_bits_[0] |= 0x00000010u;
  } else {
    _impl_._has_bits_[0] &= ~0x00000010u;
  }
  _impl_.nshead_.SetAllocated(nshead, GetArenaForAllocation());
#ifdef PROTOBUF_FORCE_COPY_DEFAULT_STRING
  if (_impl_.nshead_.IsDefault()) {
    _impl_.nshead_.Set("", GetArenaForAllocation());
  }
#endif // PROTOBUF_FORCE_COPY_DEFAULT_STRING
  // @@protoc_insertion_point(field_set_allocated:brpc.RpcDumpMeta.nshead)
}

#ifdef __GNUC__
  #pragma GCC diagnostic pop
#endif  // __GNUC__

// @@protoc_insertion_point(namespace_scope)

}  // namespace brpc

// @@protoc_insertion_point(global_scope)

#include <google/protobuf/port_undef.inc>
#endif  // GOOGLE_PROTOBUF_INCLUDED_GOOGLE_PROTOBUF_INCLUDED_brpc_2frpc_5fdump_2eproto
