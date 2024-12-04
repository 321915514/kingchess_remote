// Generated by the protocol buffer compiler.  DO NOT EDIT!
// source: brpc/trackme.proto

#ifndef GOOGLE_PROTOBUF_INCLUDED_brpc_2ftrackme_2eproto
#define GOOGLE_PROTOBUF_INCLUDED_brpc_2ftrackme_2eproto

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
#include <google/protobuf/generated_enum_reflection.h>
#include <google/protobuf/service.h>
#include <google/protobuf/unknown_field_set.h>
// @@protoc_insertion_point(includes)
#include <google/protobuf/port_def.inc>
#define PROTOBUF_INTERNAL_EXPORT_brpc_2ftrackme_2eproto
PROTOBUF_NAMESPACE_OPEN
namespace internal {
class AnyMetadata;
}  // namespace internal
PROTOBUF_NAMESPACE_CLOSE

// Internal implementation detail -- do not use these members.
struct TableStruct_brpc_2ftrackme_2eproto {
  static const uint32_t offsets[];
};
extern const ::PROTOBUF_NAMESPACE_ID::internal::DescriptorTable descriptor_table_brpc_2ftrackme_2eproto;
namespace brpc {
class TrackMeRequest;
struct TrackMeRequestDefaultTypeInternal;
extern TrackMeRequestDefaultTypeInternal _TrackMeRequest_default_instance_;
class TrackMeResponse;
struct TrackMeResponseDefaultTypeInternal;
extern TrackMeResponseDefaultTypeInternal _TrackMeResponse_default_instance_;
}  // namespace brpc
PROTOBUF_NAMESPACE_OPEN
template<> ::brpc::TrackMeRequest* Arena::CreateMaybeMessage<::brpc::TrackMeRequest>(Arena*);
template<> ::brpc::TrackMeResponse* Arena::CreateMaybeMessage<::brpc::TrackMeResponse>(Arena*);
PROTOBUF_NAMESPACE_CLOSE
namespace brpc {

enum TrackMeSeverity : int {
  TrackMeOK = 0,
  TrackMeWarning = 1,
  TrackMeFatal = 2
};
bool TrackMeSeverity_IsValid(int value);
constexpr TrackMeSeverity TrackMeSeverity_MIN = TrackMeOK;
constexpr TrackMeSeverity TrackMeSeverity_MAX = TrackMeFatal;
constexpr int TrackMeSeverity_ARRAYSIZE = TrackMeSeverity_MAX + 1;

const ::PROTOBUF_NAMESPACE_ID::EnumDescriptor* TrackMeSeverity_descriptor();
template<typename T>
inline const std::string& TrackMeSeverity_Name(T enum_t_value) {
  static_assert(::std::is_same<T, TrackMeSeverity>::value ||
    ::std::is_integral<T>::value,
    "Incorrect type passed to function TrackMeSeverity_Name.");
  return ::PROTOBUF_NAMESPACE_ID::internal::NameOfEnum(
    TrackMeSeverity_descriptor(), enum_t_value);
}
inline bool TrackMeSeverity_Parse(
    ::PROTOBUF_NAMESPACE_ID::ConstStringParam name, TrackMeSeverity* value) {
  return ::PROTOBUF_NAMESPACE_ID::internal::ParseNamedEnum<TrackMeSeverity>(
    TrackMeSeverity_descriptor(), name, value);
}
// ===================================================================

class TrackMeRequest final :
    public ::PROTOBUF_NAMESPACE_ID::Message /* @@protoc_insertion_point(class_definition:brpc.TrackMeRequest) */ {
 public:
  inline TrackMeRequest() : TrackMeRequest(nullptr) {}
  ~TrackMeRequest() override;
  explicit PROTOBUF_CONSTEXPR TrackMeRequest(::PROTOBUF_NAMESPACE_ID::internal::ConstantInitialized);

  TrackMeRequest(const TrackMeRequest& from);
  TrackMeRequest(TrackMeRequest&& from) noexcept
    : TrackMeRequest() {
    *this = ::std::move(from);
  }

  inline TrackMeRequest& operator=(const TrackMeRequest& from) {
    CopyFrom(from);
    return *this;
  }
  inline TrackMeRequest& operator=(TrackMeRequest&& from) noexcept {
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
  static const TrackMeRequest& default_instance() {
    return *internal_default_instance();
  }
  static inline const TrackMeRequest* internal_default_instance() {
    return reinterpret_cast<const TrackMeRequest*>(
               &_TrackMeRequest_default_instance_);
  }
  static constexpr int kIndexInFileMessages =
    0;

  friend void swap(TrackMeRequest& a, TrackMeRequest& b) {
    a.Swap(&b);
  }
  inline void Swap(TrackMeRequest* other) {
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
  void UnsafeArenaSwap(TrackMeRequest* other) {
    if (other == this) return;
    GOOGLE_DCHECK(GetOwningArena() == other->GetOwningArena());
    InternalSwap(other);
  }

  // implements Message ----------------------------------------------

  TrackMeRequest* New(::PROTOBUF_NAMESPACE_ID::Arena* arena = nullptr) const final {
    return CreateMaybeMessage<TrackMeRequest>(arena);
  }
  using ::PROTOBUF_NAMESPACE_ID::Message::CopyFrom;
  void CopyFrom(const TrackMeRequest& from);
  using ::PROTOBUF_NAMESPACE_ID::Message::MergeFrom;
  void MergeFrom( const TrackMeRequest& from) {
    TrackMeRequest::MergeImpl(*this, from);
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
  void InternalSwap(TrackMeRequest* other);

  private:
  friend class ::PROTOBUF_NAMESPACE_ID::internal::AnyMetadata;
  static ::PROTOBUF_NAMESPACE_ID::StringPiece FullMessageName() {
    return "brpc.TrackMeRequest";
  }
  protected:
  explicit TrackMeRequest(::PROTOBUF_NAMESPACE_ID::Arena* arena,
                       bool is_message_owned = false);
  public:

  static const ClassData _class_data_;
  const ::PROTOBUF_NAMESPACE_ID::Message::ClassData*GetClassData() const final;

  ::PROTOBUF_NAMESPACE_ID::Metadata GetMetadata() const final;

  // nested types ----------------------------------------------------

  // accessors -------------------------------------------------------

  enum : int {
    kServerAddrFieldNumber = 2,
    kRpcVersionFieldNumber = 1,
  };
  // optional string server_addr = 2;
  bool has_server_addr() const;
  private:
  bool _internal_has_server_addr() const;
  public:
  void clear_server_addr();
  const std::string& server_addr() const;
  template <typename ArgT0 = const std::string&, typename... ArgT>
  void set_server_addr(ArgT0&& arg0, ArgT... args);
  std::string* mutable_server_addr();
  PROTOBUF_NODISCARD std::string* release_server_addr();
  void set_allocated_server_addr(std::string* server_addr);
  private:
  const std::string& _internal_server_addr() const;
  inline PROTOBUF_ALWAYS_INLINE void _internal_set_server_addr(const std::string& value);
  std::string* _internal_mutable_server_addr();
  public:

  // optional int64 rpc_version = 1;
  bool has_rpc_version() const;
  private:
  bool _internal_has_rpc_version() const;
  public:
  void clear_rpc_version();
  int64_t rpc_version() const;
  void set_rpc_version(int64_t value);
  private:
  int64_t _internal_rpc_version() const;
  void _internal_set_rpc_version(int64_t value);
  public:

  // @@protoc_insertion_point(class_scope:brpc.TrackMeRequest)
 private:
  class _Internal;

  template <typename T> friend class ::PROTOBUF_NAMESPACE_ID::Arena::InternalHelper;
  typedef void InternalArenaConstructable_;
  typedef void DestructorSkippable_;
  struct Impl_ {
    ::PROTOBUF_NAMESPACE_ID::internal::HasBits<1> _has_bits_;
    mutable ::PROTOBUF_NAMESPACE_ID::internal::CachedSize _cached_size_;
    ::PROTOBUF_NAMESPACE_ID::internal::ArenaStringPtr server_addr_;
    int64_t rpc_version_;
  };
  union { Impl_ _impl_; };
  friend struct ::TableStruct_brpc_2ftrackme_2eproto;
};
// -------------------------------------------------------------------

class TrackMeResponse final :
    public ::PROTOBUF_NAMESPACE_ID::Message /* @@protoc_insertion_point(class_definition:brpc.TrackMeResponse) */ {
 public:
  inline TrackMeResponse() : TrackMeResponse(nullptr) {}
  ~TrackMeResponse() override;
  explicit PROTOBUF_CONSTEXPR TrackMeResponse(::PROTOBUF_NAMESPACE_ID::internal::ConstantInitialized);

  TrackMeResponse(const TrackMeResponse& from);
  TrackMeResponse(TrackMeResponse&& from) noexcept
    : TrackMeResponse() {
    *this = ::std::move(from);
  }

  inline TrackMeResponse& operator=(const TrackMeResponse& from) {
    CopyFrom(from);
    return *this;
  }
  inline TrackMeResponse& operator=(TrackMeResponse&& from) noexcept {
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
  static const TrackMeResponse& default_instance() {
    return *internal_default_instance();
  }
  static inline const TrackMeResponse* internal_default_instance() {
    return reinterpret_cast<const TrackMeResponse*>(
               &_TrackMeResponse_default_instance_);
  }
  static constexpr int kIndexInFileMessages =
    1;

  friend void swap(TrackMeResponse& a, TrackMeResponse& b) {
    a.Swap(&b);
  }
  inline void Swap(TrackMeResponse* other) {
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
  void UnsafeArenaSwap(TrackMeResponse* other) {
    if (other == this) return;
    GOOGLE_DCHECK(GetOwningArena() == other->GetOwningArena());
    InternalSwap(other);
  }

  // implements Message ----------------------------------------------

  TrackMeResponse* New(::PROTOBUF_NAMESPACE_ID::Arena* arena = nullptr) const final {
    return CreateMaybeMessage<TrackMeResponse>(arena);
  }
  using ::PROTOBUF_NAMESPACE_ID::Message::CopyFrom;
  void CopyFrom(const TrackMeResponse& from);
  using ::PROTOBUF_NAMESPACE_ID::Message::MergeFrom;
  void MergeFrom( const TrackMeResponse& from) {
    TrackMeResponse::MergeImpl(*this, from);
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
  void InternalSwap(TrackMeResponse* other);

  private:
  friend class ::PROTOBUF_NAMESPACE_ID::internal::AnyMetadata;
  static ::PROTOBUF_NAMESPACE_ID::StringPiece FullMessageName() {
    return "brpc.TrackMeResponse";
  }
  protected:
  explicit TrackMeResponse(::PROTOBUF_NAMESPACE_ID::Arena* arena,
                       bool is_message_owned = false);
  public:

  static const ClassData _class_data_;
  const ::PROTOBUF_NAMESPACE_ID::Message::ClassData*GetClassData() const final;

  ::PROTOBUF_NAMESPACE_ID::Metadata GetMetadata() const final;

  // nested types ----------------------------------------------------

  // accessors -------------------------------------------------------

  enum : int {
    kErrorTextFieldNumber = 2,
    kSeverityFieldNumber = 1,
    kNewIntervalFieldNumber = 3,
  };
  // optional string error_text = 2;
  bool has_error_text() const;
  private:
  bool _internal_has_error_text() const;
  public:
  void clear_error_text();
  const std::string& error_text() const;
  template <typename ArgT0 = const std::string&, typename... ArgT>
  void set_error_text(ArgT0&& arg0, ArgT... args);
  std::string* mutable_error_text();
  PROTOBUF_NODISCARD std::string* release_error_text();
  void set_allocated_error_text(std::string* error_text);
  private:
  const std::string& _internal_error_text() const;
  inline PROTOBUF_ALWAYS_INLINE void _internal_set_error_text(const std::string& value);
  std::string* _internal_mutable_error_text();
  public:

  // optional .brpc.TrackMeSeverity severity = 1;
  bool has_severity() const;
  private:
  bool _internal_has_severity() const;
  public:
  void clear_severity();
  ::brpc::TrackMeSeverity severity() const;
  void set_severity(::brpc::TrackMeSeverity value);
  private:
  ::brpc::TrackMeSeverity _internal_severity() const;
  void _internal_set_severity(::brpc::TrackMeSeverity value);
  public:

  // optional int32 new_interval = 3;
  bool has_new_interval() const;
  private:
  bool _internal_has_new_interval() const;
  public:
  void clear_new_interval();
  int32_t new_interval() const;
  void set_new_interval(int32_t value);
  private:
  int32_t _internal_new_interval() const;
  void _internal_set_new_interval(int32_t value);
  public:

  // @@protoc_insertion_point(class_scope:brpc.TrackMeResponse)
 private:
  class _Internal;

  template <typename T> friend class ::PROTOBUF_NAMESPACE_ID::Arena::InternalHelper;
  typedef void InternalArenaConstructable_;
  typedef void DestructorSkippable_;
  struct Impl_ {
    ::PROTOBUF_NAMESPACE_ID::internal::HasBits<1> _has_bits_;
    mutable ::PROTOBUF_NAMESPACE_ID::internal::CachedSize _cached_size_;
    ::PROTOBUF_NAMESPACE_ID::internal::ArenaStringPtr error_text_;
    int severity_;
    int32_t new_interval_;
  };
  union { Impl_ _impl_; };
  friend struct ::TableStruct_brpc_2ftrackme_2eproto;
};
// ===================================================================

class TrackMeService_Stub;

class TrackMeService : public ::PROTOBUF_NAMESPACE_ID::Service {
 protected:
  // This class should be treated as an abstract interface.
  inline TrackMeService() {};
 public:
  virtual ~TrackMeService();

  typedef TrackMeService_Stub Stub;

  static const ::PROTOBUF_NAMESPACE_ID::ServiceDescriptor* descriptor();

  virtual void TrackMe(::PROTOBUF_NAMESPACE_ID::RpcController* controller,
                       const ::brpc::TrackMeRequest* request,
                       ::brpc::TrackMeResponse* response,
                       ::google::protobuf::Closure* done);

  // implements Service ----------------------------------------------

  const ::PROTOBUF_NAMESPACE_ID::ServiceDescriptor* GetDescriptor();
  void CallMethod(const ::PROTOBUF_NAMESPACE_ID::MethodDescriptor* method,
                  ::PROTOBUF_NAMESPACE_ID::RpcController* controller,
                  const ::PROTOBUF_NAMESPACE_ID::Message* request,
                  ::PROTOBUF_NAMESPACE_ID::Message* response,
                  ::google::protobuf::Closure* done);
  const ::PROTOBUF_NAMESPACE_ID::Message& GetRequestPrototype(
    const ::PROTOBUF_NAMESPACE_ID::MethodDescriptor* method) const;
  const ::PROTOBUF_NAMESPACE_ID::Message& GetResponsePrototype(
    const ::PROTOBUF_NAMESPACE_ID::MethodDescriptor* method) const;

 private:
  GOOGLE_DISALLOW_EVIL_CONSTRUCTORS(TrackMeService);
};

class TrackMeService_Stub : public TrackMeService {
 public:
  TrackMeService_Stub(::PROTOBUF_NAMESPACE_ID::RpcChannel* channel);
  TrackMeService_Stub(::PROTOBUF_NAMESPACE_ID::RpcChannel* channel,
                   ::PROTOBUF_NAMESPACE_ID::Service::ChannelOwnership ownership);
  ~TrackMeService_Stub();

  inline ::PROTOBUF_NAMESPACE_ID::RpcChannel* channel() { return channel_; }

  // implements TrackMeService ------------------------------------------

  void TrackMe(::PROTOBUF_NAMESPACE_ID::RpcController* controller,
                       const ::brpc::TrackMeRequest* request,
                       ::brpc::TrackMeResponse* response,
                       ::google::protobuf::Closure* done);
 private:
  ::PROTOBUF_NAMESPACE_ID::RpcChannel* channel_;
  bool owns_channel_;
  GOOGLE_DISALLOW_EVIL_CONSTRUCTORS(TrackMeService_Stub);
};


// ===================================================================


// ===================================================================

#ifdef __GNUC__
  #pragma GCC diagnostic push
  #pragma GCC diagnostic ignored "-Wstrict-aliasing"
#endif  // __GNUC__
// TrackMeRequest

// optional int64 rpc_version = 1;
inline bool TrackMeRequest::_internal_has_rpc_version() const {
  bool value = (_impl_._has_bits_[0] & 0x00000002u) != 0;
  return value;
}
inline bool TrackMeRequest::has_rpc_version() const {
  return _internal_has_rpc_version();
}
inline void TrackMeRequest::clear_rpc_version() {
  _impl_.rpc_version_ = int64_t{0};
  _impl_._has_bits_[0] &= ~0x00000002u;
}
inline int64_t TrackMeRequest::_internal_rpc_version() const {
  return _impl_.rpc_version_;
}
inline int64_t TrackMeRequest::rpc_version() const {
  // @@protoc_insertion_point(field_get:brpc.TrackMeRequest.rpc_version)
  return _internal_rpc_version();
}
inline void TrackMeRequest::_internal_set_rpc_version(int64_t value) {
  _impl_._has_bits_[0] |= 0x00000002u;
  _impl_.rpc_version_ = value;
}
inline void TrackMeRequest::set_rpc_version(int64_t value) {
  _internal_set_rpc_version(value);
  // @@protoc_insertion_point(field_set:brpc.TrackMeRequest.rpc_version)
}

// optional string server_addr = 2;
inline bool TrackMeRequest::_internal_has_server_addr() const {
  bool value = (_impl_._has_bits_[0] & 0x00000001u) != 0;
  return value;
}
inline bool TrackMeRequest::has_server_addr() const {
  return _internal_has_server_addr();
}
inline void TrackMeRequest::clear_server_addr() {
  _impl_.server_addr_.ClearToEmpty();
  _impl_._has_bits_[0] &= ~0x00000001u;
}
inline const std::string& TrackMeRequest::server_addr() const {
  // @@protoc_insertion_point(field_get:brpc.TrackMeRequest.server_addr)
  return _internal_server_addr();
}
template <typename ArgT0, typename... ArgT>
inline PROTOBUF_ALWAYS_INLINE
void TrackMeRequest::set_server_addr(ArgT0&& arg0, ArgT... args) {
 _impl_._has_bits_[0] |= 0x00000001u;
 _impl_.server_addr_.Set(static_cast<ArgT0 &&>(arg0), args..., GetArenaForAllocation());
  // @@protoc_insertion_point(field_set:brpc.TrackMeRequest.server_addr)
}
inline std::string* TrackMeRequest::mutable_server_addr() {
  std::string* _s = _internal_mutable_server_addr();
  // @@protoc_insertion_point(field_mutable:brpc.TrackMeRequest.server_addr)
  return _s;
}
inline const std::string& TrackMeRequest::_internal_server_addr() const {
  return _impl_.server_addr_.Get();
}
inline void TrackMeRequest::_internal_set_server_addr(const std::string& value) {
  _impl_._has_bits_[0] |= 0x00000001u;
  _impl_.server_addr_.Set(value, GetArenaForAllocation());
}
inline std::string* TrackMeRequest::_internal_mutable_server_addr() {
  _impl_._has_bits_[0] |= 0x00000001u;
  return _impl_.server_addr_.Mutable(GetArenaForAllocation());
}
inline std::string* TrackMeRequest::release_server_addr() {
  // @@protoc_insertion_point(field_release:brpc.TrackMeRequest.server_addr)
  if (!_internal_has_server_addr()) {
    return nullptr;
  }
  _impl_._has_bits_[0] &= ~0x00000001u;
  auto* p = _impl_.server_addr_.Release();
#ifdef PROTOBUF_FORCE_COPY_DEFAULT_STRING
  if (_impl_.server_addr_.IsDefault()) {
    _impl_.server_addr_.Set("", GetArenaForAllocation());
  }
#endif // PROTOBUF_FORCE_COPY_DEFAULT_STRING
  return p;
}
inline void TrackMeRequest::set_allocated_server_addr(std::string* server_addr) {
  if (server_addr != nullptr) {
    _impl_._has_bits_[0] |= 0x00000001u;
  } else {
    _impl_._has_bits_[0] &= ~0x00000001u;
  }
  _impl_.server_addr_.SetAllocated(server_addr, GetArenaForAllocation());
#ifdef PROTOBUF_FORCE_COPY_DEFAULT_STRING
  if (_impl_.server_addr_.IsDefault()) {
    _impl_.server_addr_.Set("", GetArenaForAllocation());
  }
#endif // PROTOBUF_FORCE_COPY_DEFAULT_STRING
  // @@protoc_insertion_point(field_set_allocated:brpc.TrackMeRequest.server_addr)
}

// -------------------------------------------------------------------

// TrackMeResponse

// optional .brpc.TrackMeSeverity severity = 1;
inline bool TrackMeResponse::_internal_has_severity() const {
  bool value = (_impl_._has_bits_[0] & 0x00000002u) != 0;
  return value;
}
inline bool TrackMeResponse::has_severity() const {
  return _internal_has_severity();
}
inline void TrackMeResponse::clear_severity() {
  _impl_.severity_ = 0;
  _impl_._has_bits_[0] &= ~0x00000002u;
}
inline ::brpc::TrackMeSeverity TrackMeResponse::_internal_severity() const {
  return static_cast< ::brpc::TrackMeSeverity >(_impl_.severity_);
}
inline ::brpc::TrackMeSeverity TrackMeResponse::severity() const {
  // @@protoc_insertion_point(field_get:brpc.TrackMeResponse.severity)
  return _internal_severity();
}
inline void TrackMeResponse::_internal_set_severity(::brpc::TrackMeSeverity value) {
  assert(::brpc::TrackMeSeverity_IsValid(value));
  _impl_._has_bits_[0] |= 0x00000002u;
  _impl_.severity_ = value;
}
inline void TrackMeResponse::set_severity(::brpc::TrackMeSeverity value) {
  _internal_set_severity(value);
  // @@protoc_insertion_point(field_set:brpc.TrackMeResponse.severity)
}

// optional string error_text = 2;
inline bool TrackMeResponse::_internal_has_error_text() const {
  bool value = (_impl_._has_bits_[0] & 0x00000001u) != 0;
  return value;
}
inline bool TrackMeResponse::has_error_text() const {
  return _internal_has_error_text();
}
inline void TrackMeResponse::clear_error_text() {
  _impl_.error_text_.ClearToEmpty();
  _impl_._has_bits_[0] &= ~0x00000001u;
}
inline const std::string& TrackMeResponse::error_text() const {
  // @@protoc_insertion_point(field_get:brpc.TrackMeResponse.error_text)
  return _internal_error_text();
}
template <typename ArgT0, typename... ArgT>
inline PROTOBUF_ALWAYS_INLINE
void TrackMeResponse::set_error_text(ArgT0&& arg0, ArgT... args) {
 _impl_._has_bits_[0] |= 0x00000001u;
 _impl_.error_text_.Set(static_cast<ArgT0 &&>(arg0), args..., GetArenaForAllocation());
  // @@protoc_insertion_point(field_set:brpc.TrackMeResponse.error_text)
}
inline std::string* TrackMeResponse::mutable_error_text() {
  std::string* _s = _internal_mutable_error_text();
  // @@protoc_insertion_point(field_mutable:brpc.TrackMeResponse.error_text)
  return _s;
}
inline const std::string& TrackMeResponse::_internal_error_text() const {
  return _impl_.error_text_.Get();
}
inline void TrackMeResponse::_internal_set_error_text(const std::string& value) {
  _impl_._has_bits_[0] |= 0x00000001u;
  _impl_.error_text_.Set(value, GetArenaForAllocation());
}
inline std::string* TrackMeResponse::_internal_mutable_error_text() {
  _impl_._has_bits_[0] |= 0x00000001u;
  return _impl_.error_text_.Mutable(GetArenaForAllocation());
}
inline std::string* TrackMeResponse::release_error_text() {
  // @@protoc_insertion_point(field_release:brpc.TrackMeResponse.error_text)
  if (!_internal_has_error_text()) {
    return nullptr;
  }
  _impl_._has_bits_[0] &= ~0x00000001u;
  auto* p = _impl_.error_text_.Release();
#ifdef PROTOBUF_FORCE_COPY_DEFAULT_STRING
  if (_impl_.error_text_.IsDefault()) {
    _impl_.error_text_.Set("", GetArenaForAllocation());
  }
#endif // PROTOBUF_FORCE_COPY_DEFAULT_STRING
  return p;
}
inline void TrackMeResponse::set_allocated_error_text(std::string* error_text) {
  if (error_text != nullptr) {
    _impl_._has_bits_[0] |= 0x00000001u;
  } else {
    _impl_._has_bits_[0] &= ~0x00000001u;
  }
  _impl_.error_text_.SetAllocated(error_text, GetArenaForAllocation());
#ifdef PROTOBUF_FORCE_COPY_DEFAULT_STRING
  if (_impl_.error_text_.IsDefault()) {
    _impl_.error_text_.Set("", GetArenaForAllocation());
  }
#endif // PROTOBUF_FORCE_COPY_DEFAULT_STRING
  // @@protoc_insertion_point(field_set_allocated:brpc.TrackMeResponse.error_text)
}

// optional int32 new_interval = 3;
inline bool TrackMeResponse::_internal_has_new_interval() const {
  bool value = (_impl_._has_bits_[0] & 0x00000004u) != 0;
  return value;
}
inline bool TrackMeResponse::has_new_interval() const {
  return _internal_has_new_interval();
}
inline void TrackMeResponse::clear_new_interval() {
  _impl_.new_interval_ = 0;
  _impl_._has_bits_[0] &= ~0x00000004u;
}
inline int32_t TrackMeResponse::_internal_new_interval() const {
  return _impl_.new_interval_;
}
inline int32_t TrackMeResponse::new_interval() const {
  // @@protoc_insertion_point(field_get:brpc.TrackMeResponse.new_interval)
  return _internal_new_interval();
}
inline void TrackMeResponse::_internal_set_new_interval(int32_t value) {
  _impl_._has_bits_[0] |= 0x00000004u;
  _impl_.new_interval_ = value;
}
inline void TrackMeResponse::set_new_interval(int32_t value) {
  _internal_set_new_interval(value);
  // @@protoc_insertion_point(field_set:brpc.TrackMeResponse.new_interval)
}

#ifdef __GNUC__
  #pragma GCC diagnostic pop
#endif  // __GNUC__
// -------------------------------------------------------------------


// @@protoc_insertion_point(namespace_scope)

}  // namespace brpc

PROTOBUF_NAMESPACE_OPEN

template <> struct is_proto_enum< ::brpc::TrackMeSeverity> : ::std::true_type {};
template <>
inline const EnumDescriptor* GetEnumDescriptor< ::brpc::TrackMeSeverity>() {
  return ::brpc::TrackMeSeverity_descriptor();
}

PROTOBUF_NAMESPACE_CLOSE

// @@protoc_insertion_point(global_scope)

#include <google/protobuf/port_undef.inc>
#endif  // GOOGLE_PROTOBUF_INCLUDED_GOOGLE_PROTOBUF_INCLUDED_brpc_2ftrackme_2eproto
