// Generated by the protocol buffer compiler.  DO NOT EDIT!
// source: brpc/get_js.proto

#ifndef GOOGLE_PROTOBUF_INCLUDED_brpc_2fget_5fjs_2eproto
#define GOOGLE_PROTOBUF_INCLUDED_brpc_2fget_5fjs_2eproto

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
#include <google/protobuf/generated_message_bases.h>
#include <google/protobuf/generated_message_util.h>
#include <google/protobuf/metadata_lite.h>
#include <google/protobuf/generated_message_reflection.h>
#include <google/protobuf/message.h>
#include <google/protobuf/repeated_field.h>  // IWYU pragma: export
#include <google/protobuf/extension_set.h>  // IWYU pragma: export
#include <google/protobuf/service.h>
#include <google/protobuf/unknown_field_set.h>
// @@protoc_insertion_point(includes)
#include <google/protobuf/port_def.inc>
#define PROTOBUF_INTERNAL_EXPORT_brpc_2fget_5fjs_2eproto
PROTOBUF_NAMESPACE_OPEN
namespace internal {
class AnyMetadata;
}  // namespace internal
PROTOBUF_NAMESPACE_CLOSE

// Internal implementation detail -- do not use these members.
struct TableStruct_brpc_2fget_5fjs_2eproto {
  static const uint32_t offsets[];
};
extern const ::PROTOBUF_NAMESPACE_ID::internal::DescriptorTable descriptor_table_brpc_2fget_5fjs_2eproto;
namespace brpc {
class GetJsRequest;
struct GetJsRequestDefaultTypeInternal;
extern GetJsRequestDefaultTypeInternal _GetJsRequest_default_instance_;
class GetJsResponse;
struct GetJsResponseDefaultTypeInternal;
extern GetJsResponseDefaultTypeInternal _GetJsResponse_default_instance_;
}  // namespace brpc
PROTOBUF_NAMESPACE_OPEN
template<> ::brpc::GetJsRequest* Arena::CreateMaybeMessage<::brpc::GetJsRequest>(Arena*);
template<> ::brpc::GetJsResponse* Arena::CreateMaybeMessage<::brpc::GetJsResponse>(Arena*);
PROTOBUF_NAMESPACE_CLOSE
namespace brpc {

// ===================================================================

class GetJsRequest final :
    public ::PROTOBUF_NAMESPACE_ID::internal::ZeroFieldsBase /* @@protoc_insertion_point(class_definition:brpc.GetJsRequest) */ {
 public:
  inline GetJsRequest() : GetJsRequest(nullptr) {}
  explicit PROTOBUF_CONSTEXPR GetJsRequest(::PROTOBUF_NAMESPACE_ID::internal::ConstantInitialized);

  GetJsRequest(const GetJsRequest& from);
  GetJsRequest(GetJsRequest&& from) noexcept
    : GetJsRequest() {
    *this = ::std::move(from);
  }

  inline GetJsRequest& operator=(const GetJsRequest& from) {
    CopyFrom(from);
    return *this;
  }
  inline GetJsRequest& operator=(GetJsRequest&& from) noexcept {
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
  static const GetJsRequest& default_instance() {
    return *internal_default_instance();
  }
  static inline const GetJsRequest* internal_default_instance() {
    return reinterpret_cast<const GetJsRequest*>(
               &_GetJsRequest_default_instance_);
  }
  static constexpr int kIndexInFileMessages =
    0;

  friend void swap(GetJsRequest& a, GetJsRequest& b) {
    a.Swap(&b);
  }
  inline void Swap(GetJsRequest* other) {
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
  void UnsafeArenaSwap(GetJsRequest* other) {
    if (other == this) return;
    GOOGLE_DCHECK(GetOwningArena() == other->GetOwningArena());
    InternalSwap(other);
  }

  // implements Message ----------------------------------------------

  GetJsRequest* New(::PROTOBUF_NAMESPACE_ID::Arena* arena = nullptr) const final {
    return CreateMaybeMessage<GetJsRequest>(arena);
  }
  using ::PROTOBUF_NAMESPACE_ID::internal::ZeroFieldsBase::CopyFrom;
  inline void CopyFrom(const GetJsRequest& from) {
    ::PROTOBUF_NAMESPACE_ID::internal::ZeroFieldsBase::CopyImpl(*this, from);
  }
  using ::PROTOBUF_NAMESPACE_ID::internal::ZeroFieldsBase::MergeFrom;
  void MergeFrom(const GetJsRequest& from) {
    ::PROTOBUF_NAMESPACE_ID::internal::ZeroFieldsBase::MergeImpl(*this, from);
  }
  public:

  private:
  friend class ::PROTOBUF_NAMESPACE_ID::internal::AnyMetadata;
  static ::PROTOBUF_NAMESPACE_ID::StringPiece FullMessageName() {
    return "brpc.GetJsRequest";
  }
  protected:
  explicit GetJsRequest(::PROTOBUF_NAMESPACE_ID::Arena* arena,
                       bool is_message_owned = false);
  public:

  static const ClassData _class_data_;
  const ::PROTOBUF_NAMESPACE_ID::Message::ClassData*GetClassData() const final;

  ::PROTOBUF_NAMESPACE_ID::Metadata GetMetadata() const final;

  // nested types ----------------------------------------------------

  // accessors -------------------------------------------------------

  // @@protoc_insertion_point(class_scope:brpc.GetJsRequest)
 private:
  class _Internal;

  template <typename T> friend class ::PROTOBUF_NAMESPACE_ID::Arena::InternalHelper;
  typedef void InternalArenaConstructable_;
  typedef void DestructorSkippable_;
  struct Impl_ {
  };
  friend struct ::TableStruct_brpc_2fget_5fjs_2eproto;
};
// -------------------------------------------------------------------

class GetJsResponse final :
    public ::PROTOBUF_NAMESPACE_ID::internal::ZeroFieldsBase /* @@protoc_insertion_point(class_definition:brpc.GetJsResponse) */ {
 public:
  inline GetJsResponse() : GetJsResponse(nullptr) {}
  explicit PROTOBUF_CONSTEXPR GetJsResponse(::PROTOBUF_NAMESPACE_ID::internal::ConstantInitialized);

  GetJsResponse(const GetJsResponse& from);
  GetJsResponse(GetJsResponse&& from) noexcept
    : GetJsResponse() {
    *this = ::std::move(from);
  }

  inline GetJsResponse& operator=(const GetJsResponse& from) {
    CopyFrom(from);
    return *this;
  }
  inline GetJsResponse& operator=(GetJsResponse&& from) noexcept {
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
  static const GetJsResponse& default_instance() {
    return *internal_default_instance();
  }
  static inline const GetJsResponse* internal_default_instance() {
    return reinterpret_cast<const GetJsResponse*>(
               &_GetJsResponse_default_instance_);
  }
  static constexpr int kIndexInFileMessages =
    1;

  friend void swap(GetJsResponse& a, GetJsResponse& b) {
    a.Swap(&b);
  }
  inline void Swap(GetJsResponse* other) {
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
  void UnsafeArenaSwap(GetJsResponse* other) {
    if (other == this) return;
    GOOGLE_DCHECK(GetOwningArena() == other->GetOwningArena());
    InternalSwap(other);
  }

  // implements Message ----------------------------------------------

  GetJsResponse* New(::PROTOBUF_NAMESPACE_ID::Arena* arena = nullptr) const final {
    return CreateMaybeMessage<GetJsResponse>(arena);
  }
  using ::PROTOBUF_NAMESPACE_ID::internal::ZeroFieldsBase::CopyFrom;
  inline void CopyFrom(const GetJsResponse& from) {
    ::PROTOBUF_NAMESPACE_ID::internal::ZeroFieldsBase::CopyImpl(*this, from);
  }
  using ::PROTOBUF_NAMESPACE_ID::internal::ZeroFieldsBase::MergeFrom;
  void MergeFrom(const GetJsResponse& from) {
    ::PROTOBUF_NAMESPACE_ID::internal::ZeroFieldsBase::MergeImpl(*this, from);
  }
  public:

  private:
  friend class ::PROTOBUF_NAMESPACE_ID::internal::AnyMetadata;
  static ::PROTOBUF_NAMESPACE_ID::StringPiece FullMessageName() {
    return "brpc.GetJsResponse";
  }
  protected:
  explicit GetJsResponse(::PROTOBUF_NAMESPACE_ID::Arena* arena,
                       bool is_message_owned = false);
  public:

  static const ClassData _class_data_;
  const ::PROTOBUF_NAMESPACE_ID::Message::ClassData*GetClassData() const final;

  ::PROTOBUF_NAMESPACE_ID::Metadata GetMetadata() const final;

  // nested types ----------------------------------------------------

  // accessors -------------------------------------------------------

  // @@protoc_insertion_point(class_scope:brpc.GetJsResponse)
 private:
  class _Internal;

  template <typename T> friend class ::PROTOBUF_NAMESPACE_ID::Arena::InternalHelper;
  typedef void InternalArenaConstructable_;
  typedef void DestructorSkippable_;
  struct Impl_ {
  };
  friend struct ::TableStruct_brpc_2fget_5fjs_2eproto;
};
// ===================================================================

class js_Stub;

class js : public ::PROTOBUF_NAMESPACE_ID::Service {
 protected:
  // This class should be treated as an abstract interface.
  inline js() {};
 public:
  virtual ~js();

  typedef js_Stub Stub;

  static const ::PROTOBUF_NAMESPACE_ID::ServiceDescriptor* descriptor();

  virtual void sorttable(::PROTOBUF_NAMESPACE_ID::RpcController* controller,
                       const ::brpc::GetJsRequest* request,
                       ::brpc::GetJsResponse* response,
                       ::google::protobuf::Closure* done);
  virtual void jquery_min(::PROTOBUF_NAMESPACE_ID::RpcController* controller,
                       const ::brpc::GetJsRequest* request,
                       ::brpc::GetJsResponse* response,
                       ::google::protobuf::Closure* done);
  virtual void flot_min(::PROTOBUF_NAMESPACE_ID::RpcController* controller,
                       const ::brpc::GetJsRequest* request,
                       ::brpc::GetJsResponse* response,
                       ::google::protobuf::Closure* done);
  virtual void viz_min(::PROTOBUF_NAMESPACE_ID::RpcController* controller,
                       const ::brpc::GetJsRequest* request,
                       ::brpc::GetJsResponse* response,
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
  GOOGLE_DISALLOW_EVIL_CONSTRUCTORS(js);
};

class js_Stub : public js {
 public:
  js_Stub(::PROTOBUF_NAMESPACE_ID::RpcChannel* channel);
  js_Stub(::PROTOBUF_NAMESPACE_ID::RpcChannel* channel,
                   ::PROTOBUF_NAMESPACE_ID::Service::ChannelOwnership ownership);
  ~js_Stub();

  inline ::PROTOBUF_NAMESPACE_ID::RpcChannel* channel() { return channel_; }

  // implements js ------------------------------------------

  void sorttable(::PROTOBUF_NAMESPACE_ID::RpcController* controller,
                       const ::brpc::GetJsRequest* request,
                       ::brpc::GetJsResponse* response,
                       ::google::protobuf::Closure* done);
  void jquery_min(::PROTOBUF_NAMESPACE_ID::RpcController* controller,
                       const ::brpc::GetJsRequest* request,
                       ::brpc::GetJsResponse* response,
                       ::google::protobuf::Closure* done);
  void flot_min(::PROTOBUF_NAMESPACE_ID::RpcController* controller,
                       const ::brpc::GetJsRequest* request,
                       ::brpc::GetJsResponse* response,
                       ::google::protobuf::Closure* done);
  void viz_min(::PROTOBUF_NAMESPACE_ID::RpcController* controller,
                       const ::brpc::GetJsRequest* request,
                       ::brpc::GetJsResponse* response,
                       ::google::protobuf::Closure* done);
 private:
  ::PROTOBUF_NAMESPACE_ID::RpcChannel* channel_;
  bool owns_channel_;
  GOOGLE_DISALLOW_EVIL_CONSTRUCTORS(js_Stub);
};


// ===================================================================


// ===================================================================

#ifdef __GNUC__
  #pragma GCC diagnostic push
  #pragma GCC diagnostic ignored "-Wstrict-aliasing"
#endif  // __GNUC__
// GetJsRequest

// -------------------------------------------------------------------

// GetJsResponse

#ifdef __GNUC__
  #pragma GCC diagnostic pop
#endif  // __GNUC__
// -------------------------------------------------------------------


// @@protoc_insertion_point(namespace_scope)

}  // namespace brpc

// @@protoc_insertion_point(global_scope)

#include <google/protobuf/port_undef.inc>
#endif  // GOOGLE_PROTOBUF_INCLUDED_GOOGLE_PROTOBUF_INCLUDED_brpc_2fget_5fjs_2eproto
