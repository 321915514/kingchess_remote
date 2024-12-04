// Generated by the protocol buffer compiler.  DO NOT EDIT!
// source: brpc/streaming_rpc_meta.proto

#ifndef GOOGLE_PROTOBUF_INCLUDED_brpc_2fstreaming_5frpc_5fmeta_2eproto
#define GOOGLE_PROTOBUF_INCLUDED_brpc_2fstreaming_5frpc_5fmeta_2eproto

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
#include <google/protobuf/unknown_field_set.h>
// @@protoc_insertion_point(includes)
#include <google/protobuf/port_def.inc>
#define PROTOBUF_INTERNAL_EXPORT_brpc_2fstreaming_5frpc_5fmeta_2eproto
PROTOBUF_NAMESPACE_OPEN
namespace internal {
class AnyMetadata;
}  // namespace internal
PROTOBUF_NAMESPACE_CLOSE

// Internal implementation detail -- do not use these members.
struct TableStruct_brpc_2fstreaming_5frpc_5fmeta_2eproto {
  static const uint32_t offsets[];
};
extern const ::PROTOBUF_NAMESPACE_ID::internal::DescriptorTable descriptor_table_brpc_2fstreaming_5frpc_5fmeta_2eproto;
namespace brpc {
class Feedback;
struct FeedbackDefaultTypeInternal;
extern FeedbackDefaultTypeInternal _Feedback_default_instance_;
class StreamFrameMeta;
struct StreamFrameMetaDefaultTypeInternal;
extern StreamFrameMetaDefaultTypeInternal _StreamFrameMeta_default_instance_;
class StreamSettings;
struct StreamSettingsDefaultTypeInternal;
extern StreamSettingsDefaultTypeInternal _StreamSettings_default_instance_;
}  // namespace brpc
PROTOBUF_NAMESPACE_OPEN
template<> ::brpc::Feedback* Arena::CreateMaybeMessage<::brpc::Feedback>(Arena*);
template<> ::brpc::StreamFrameMeta* Arena::CreateMaybeMessage<::brpc::StreamFrameMeta>(Arena*);
template<> ::brpc::StreamSettings* Arena::CreateMaybeMessage<::brpc::StreamSettings>(Arena*);
PROTOBUF_NAMESPACE_CLOSE
namespace brpc {

enum FrameType : int {
  FRAME_TYPE_UNKNOWN = 0,
  FRAME_TYPE_RST = 1,
  FRAME_TYPE_CLOSE = 2,
  FRAME_TYPE_DATA = 3,
  FRAME_TYPE_FEEDBACK = 4
};
bool FrameType_IsValid(int value);
constexpr FrameType FrameType_MIN = FRAME_TYPE_UNKNOWN;
constexpr FrameType FrameType_MAX = FRAME_TYPE_FEEDBACK;
constexpr int FrameType_ARRAYSIZE = FrameType_MAX + 1;

const ::PROTOBUF_NAMESPACE_ID::EnumDescriptor* FrameType_descriptor();
template<typename T>
inline const std::string& FrameType_Name(T enum_t_value) {
  static_assert(::std::is_same<T, FrameType>::value ||
    ::std::is_integral<T>::value,
    "Incorrect type passed to function FrameType_Name.");
  return ::PROTOBUF_NAMESPACE_ID::internal::NameOfEnum(
    FrameType_descriptor(), enum_t_value);
}
inline bool FrameType_Parse(
    ::PROTOBUF_NAMESPACE_ID::ConstStringParam name, FrameType* value) {
  return ::PROTOBUF_NAMESPACE_ID::internal::ParseNamedEnum<FrameType>(
    FrameType_descriptor(), name, value);
}
// ===================================================================

class StreamSettings final :
    public ::PROTOBUF_NAMESPACE_ID::Message /* @@protoc_insertion_point(class_definition:brpc.StreamSettings) */ {
 public:
  inline StreamSettings() : StreamSettings(nullptr) {}
  ~StreamSettings() override;
  explicit PROTOBUF_CONSTEXPR StreamSettings(::PROTOBUF_NAMESPACE_ID::internal::ConstantInitialized);

  StreamSettings(const StreamSettings& from);
  StreamSettings(StreamSettings&& from) noexcept
    : StreamSettings() {
    *this = ::std::move(from);
  }

  inline StreamSettings& operator=(const StreamSettings& from) {
    CopyFrom(from);
    return *this;
  }
  inline StreamSettings& operator=(StreamSettings&& from) noexcept {
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
  static const StreamSettings& default_instance() {
    return *internal_default_instance();
  }
  static inline const StreamSettings* internal_default_instance() {
    return reinterpret_cast<const StreamSettings*>(
               &_StreamSettings_default_instance_);
  }
  static constexpr int kIndexInFileMessages =
    0;

  friend void swap(StreamSettings& a, StreamSettings& b) {
    a.Swap(&b);
  }
  inline void Swap(StreamSettings* other) {
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
  void UnsafeArenaSwap(StreamSettings* other) {
    if (other == this) return;
    GOOGLE_DCHECK(GetOwningArena() == other->GetOwningArena());
    InternalSwap(other);
  }

  // implements Message ----------------------------------------------

  StreamSettings* New(::PROTOBUF_NAMESPACE_ID::Arena* arena = nullptr) const final {
    return CreateMaybeMessage<StreamSettings>(arena);
  }
  using ::PROTOBUF_NAMESPACE_ID::Message::CopyFrom;
  void CopyFrom(const StreamSettings& from);
  using ::PROTOBUF_NAMESPACE_ID::Message::MergeFrom;
  void MergeFrom( const StreamSettings& from) {
    StreamSettings::MergeImpl(*this, from);
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
  void InternalSwap(StreamSettings* other);

  private:
  friend class ::PROTOBUF_NAMESPACE_ID::internal::AnyMetadata;
  static ::PROTOBUF_NAMESPACE_ID::StringPiece FullMessageName() {
    return "brpc.StreamSettings";
  }
  protected:
  explicit StreamSettings(::PROTOBUF_NAMESPACE_ID::Arena* arena,
                       bool is_message_owned = false);
  public:

  static const ClassData _class_data_;
  const ::PROTOBUF_NAMESPACE_ID::Message::ClassData*GetClassData() const final;

  ::PROTOBUF_NAMESPACE_ID::Metadata GetMetadata() const final;

  // nested types ----------------------------------------------------

  // accessors -------------------------------------------------------

  enum : int {
    kStreamIdFieldNumber = 1,
    kNeedFeedbackFieldNumber = 2,
    kWritableFieldNumber = 3,
  };
  // required int64 stream_id = 1;
  bool has_stream_id() const;
  private:
  bool _internal_has_stream_id() const;
  public:
  void clear_stream_id();
  int64_t stream_id() const;
  void set_stream_id(int64_t value);
  private:
  int64_t _internal_stream_id() const;
  void _internal_set_stream_id(int64_t value);
  public:

  // optional bool need_feedback = 2 [default = false];
  bool has_need_feedback() const;
  private:
  bool _internal_has_need_feedback() const;
  public:
  void clear_need_feedback();
  bool need_feedback() const;
  void set_need_feedback(bool value);
  private:
  bool _internal_need_feedback() const;
  void _internal_set_need_feedback(bool value);
  public:

  // optional bool writable = 3 [default = false];
  bool has_writable() const;
  private:
  bool _internal_has_writable() const;
  public:
  void clear_writable();
  bool writable() const;
  void set_writable(bool value);
  private:
  bool _internal_writable() const;
  void _internal_set_writable(bool value);
  public:

  // @@protoc_insertion_point(class_scope:brpc.StreamSettings)
 private:
  class _Internal;

  template <typename T> friend class ::PROTOBUF_NAMESPACE_ID::Arena::InternalHelper;
  typedef void InternalArenaConstructable_;
  typedef void DestructorSkippable_;
  struct Impl_ {
    ::PROTOBUF_NAMESPACE_ID::internal::HasBits<1> _has_bits_;
    mutable ::PROTOBUF_NAMESPACE_ID::internal::CachedSize _cached_size_;
    int64_t stream_id_;
    bool need_feedback_;
    bool writable_;
  };
  union { Impl_ _impl_; };
  friend struct ::TableStruct_brpc_2fstreaming_5frpc_5fmeta_2eproto;
};
// -------------------------------------------------------------------

class StreamFrameMeta final :
    public ::PROTOBUF_NAMESPACE_ID::Message /* @@protoc_insertion_point(class_definition:brpc.StreamFrameMeta) */ {
 public:
  inline StreamFrameMeta() : StreamFrameMeta(nullptr) {}
  ~StreamFrameMeta() override;
  explicit PROTOBUF_CONSTEXPR StreamFrameMeta(::PROTOBUF_NAMESPACE_ID::internal::ConstantInitialized);

  StreamFrameMeta(const StreamFrameMeta& from);
  StreamFrameMeta(StreamFrameMeta&& from) noexcept
    : StreamFrameMeta() {
    *this = ::std::move(from);
  }

  inline StreamFrameMeta& operator=(const StreamFrameMeta& from) {
    CopyFrom(from);
    return *this;
  }
  inline StreamFrameMeta& operator=(StreamFrameMeta&& from) noexcept {
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
  static const StreamFrameMeta& default_instance() {
    return *internal_default_instance();
  }
  static inline const StreamFrameMeta* internal_default_instance() {
    return reinterpret_cast<const StreamFrameMeta*>(
               &_StreamFrameMeta_default_instance_);
  }
  static constexpr int kIndexInFileMessages =
    1;

  friend void swap(StreamFrameMeta& a, StreamFrameMeta& b) {
    a.Swap(&b);
  }
  inline void Swap(StreamFrameMeta* other) {
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
  void UnsafeArenaSwap(StreamFrameMeta* other) {
    if (other == this) return;
    GOOGLE_DCHECK(GetOwningArena() == other->GetOwningArena());
    InternalSwap(other);
  }

  // implements Message ----------------------------------------------

  StreamFrameMeta* New(::PROTOBUF_NAMESPACE_ID::Arena* arena = nullptr) const final {
    return CreateMaybeMessage<StreamFrameMeta>(arena);
  }
  using ::PROTOBUF_NAMESPACE_ID::Message::CopyFrom;
  void CopyFrom(const StreamFrameMeta& from);
  using ::PROTOBUF_NAMESPACE_ID::Message::MergeFrom;
  void MergeFrom( const StreamFrameMeta& from) {
    StreamFrameMeta::MergeImpl(*this, from);
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
  void InternalSwap(StreamFrameMeta* other);

  private:
  friend class ::PROTOBUF_NAMESPACE_ID::internal::AnyMetadata;
  static ::PROTOBUF_NAMESPACE_ID::StringPiece FullMessageName() {
    return "brpc.StreamFrameMeta";
  }
  protected:
  explicit StreamFrameMeta(::PROTOBUF_NAMESPACE_ID::Arena* arena,
                       bool is_message_owned = false);
  public:

  static const ClassData _class_data_;
  const ::PROTOBUF_NAMESPACE_ID::Message::ClassData*GetClassData() const final;

  ::PROTOBUF_NAMESPACE_ID::Metadata GetMetadata() const final;

  // nested types ----------------------------------------------------

  // accessors -------------------------------------------------------

  enum : int {
    kFeedbackFieldNumber = 5,
    kStreamIdFieldNumber = 1,
    kSourceStreamIdFieldNumber = 2,
    kFrameTypeFieldNumber = 3,
    kHasContinuationFieldNumber = 4,
  };
  // optional .brpc.Feedback feedback = 5;
  bool has_feedback() const;
  private:
  bool _internal_has_feedback() const;
  public:
  void clear_feedback();
  const ::brpc::Feedback& feedback() const;
  PROTOBUF_NODISCARD ::brpc::Feedback* release_feedback();
  ::brpc::Feedback* mutable_feedback();
  void set_allocated_feedback(::brpc::Feedback* feedback);
  private:
  const ::brpc::Feedback& _internal_feedback() const;
  ::brpc::Feedback* _internal_mutable_feedback();
  public:
  void unsafe_arena_set_allocated_feedback(
      ::brpc::Feedback* feedback);
  ::brpc::Feedback* unsafe_arena_release_feedback();

  // required int64 stream_id = 1;
  bool has_stream_id() const;
  private:
  bool _internal_has_stream_id() const;
  public:
  void clear_stream_id();
  int64_t stream_id() const;
  void set_stream_id(int64_t value);
  private:
  int64_t _internal_stream_id() const;
  void _internal_set_stream_id(int64_t value);
  public:

  // optional int64 source_stream_id = 2;
  bool has_source_stream_id() const;
  private:
  bool _internal_has_source_stream_id() const;
  public:
  void clear_source_stream_id();
  int64_t source_stream_id() const;
  void set_source_stream_id(int64_t value);
  private:
  int64_t _internal_source_stream_id() const;
  void _internal_set_source_stream_id(int64_t value);
  public:

  // optional .brpc.FrameType frame_type = 3;
  bool has_frame_type() const;
  private:
  bool _internal_has_frame_type() const;
  public:
  void clear_frame_type();
  ::brpc::FrameType frame_type() const;
  void set_frame_type(::brpc::FrameType value);
  private:
  ::brpc::FrameType _internal_frame_type() const;
  void _internal_set_frame_type(::brpc::FrameType value);
  public:

  // optional bool has_continuation = 4;
  bool has_has_continuation() const;
  private:
  bool _internal_has_has_continuation() const;
  public:
  void clear_has_continuation();
  bool has_continuation() const;
  void set_has_continuation(bool value);
  private:
  bool _internal_has_continuation() const;
  void _internal_set_has_continuation(bool value);
  public:

  // @@protoc_insertion_point(class_scope:brpc.StreamFrameMeta)
 private:
  class _Internal;

  template <typename T> friend class ::PROTOBUF_NAMESPACE_ID::Arena::InternalHelper;
  typedef void InternalArenaConstructable_;
  typedef void DestructorSkippable_;
  struct Impl_ {
    ::PROTOBUF_NAMESPACE_ID::internal::HasBits<1> _has_bits_;
    mutable ::PROTOBUF_NAMESPACE_ID::internal::CachedSize _cached_size_;
    ::brpc::Feedback* feedback_;
    int64_t stream_id_;
    int64_t source_stream_id_;
    int frame_type_;
    bool has_continuation_;
  };
  union { Impl_ _impl_; };
  friend struct ::TableStruct_brpc_2fstreaming_5frpc_5fmeta_2eproto;
};
// -------------------------------------------------------------------

class Feedback final :
    public ::PROTOBUF_NAMESPACE_ID::Message /* @@protoc_insertion_point(class_definition:brpc.Feedback) */ {
 public:
  inline Feedback() : Feedback(nullptr) {}
  ~Feedback() override;
  explicit PROTOBUF_CONSTEXPR Feedback(::PROTOBUF_NAMESPACE_ID::internal::ConstantInitialized);

  Feedback(const Feedback& from);
  Feedback(Feedback&& from) noexcept
    : Feedback() {
    *this = ::std::move(from);
  }

  inline Feedback& operator=(const Feedback& from) {
    CopyFrom(from);
    return *this;
  }
  inline Feedback& operator=(Feedback&& from) noexcept {
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
  static const Feedback& default_instance() {
    return *internal_default_instance();
  }
  static inline const Feedback* internal_default_instance() {
    return reinterpret_cast<const Feedback*>(
               &_Feedback_default_instance_);
  }
  static constexpr int kIndexInFileMessages =
    2;

  friend void swap(Feedback& a, Feedback& b) {
    a.Swap(&b);
  }
  inline void Swap(Feedback* other) {
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
  void UnsafeArenaSwap(Feedback* other) {
    if (other == this) return;
    GOOGLE_DCHECK(GetOwningArena() == other->GetOwningArena());
    InternalSwap(other);
  }

  // implements Message ----------------------------------------------

  Feedback* New(::PROTOBUF_NAMESPACE_ID::Arena* arena = nullptr) const final {
    return CreateMaybeMessage<Feedback>(arena);
  }
  using ::PROTOBUF_NAMESPACE_ID::Message::CopyFrom;
  void CopyFrom(const Feedback& from);
  using ::PROTOBUF_NAMESPACE_ID::Message::MergeFrom;
  void MergeFrom( const Feedback& from) {
    Feedback::MergeImpl(*this, from);
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
  void InternalSwap(Feedback* other);

  private:
  friend class ::PROTOBUF_NAMESPACE_ID::internal::AnyMetadata;
  static ::PROTOBUF_NAMESPACE_ID::StringPiece FullMessageName() {
    return "brpc.Feedback";
  }
  protected:
  explicit Feedback(::PROTOBUF_NAMESPACE_ID::Arena* arena,
                       bool is_message_owned = false);
  public:

  static const ClassData _class_data_;
  const ::PROTOBUF_NAMESPACE_ID::Message::ClassData*GetClassData() const final;

  ::PROTOBUF_NAMESPACE_ID::Metadata GetMetadata() const final;

  // nested types ----------------------------------------------------

  // accessors -------------------------------------------------------

  enum : int {
    kConsumedSizeFieldNumber = 1,
  };
  // optional int64 consumed_size = 1;
  bool has_consumed_size() const;
  private:
  bool _internal_has_consumed_size() const;
  public:
  void clear_consumed_size();
  int64_t consumed_size() const;
  void set_consumed_size(int64_t value);
  private:
  int64_t _internal_consumed_size() const;
  void _internal_set_consumed_size(int64_t value);
  public:

  // @@protoc_insertion_point(class_scope:brpc.Feedback)
 private:
  class _Internal;

  template <typename T> friend class ::PROTOBUF_NAMESPACE_ID::Arena::InternalHelper;
  typedef void InternalArenaConstructable_;
  typedef void DestructorSkippable_;
  struct Impl_ {
    ::PROTOBUF_NAMESPACE_ID::internal::HasBits<1> _has_bits_;
    mutable ::PROTOBUF_NAMESPACE_ID::internal::CachedSize _cached_size_;
    int64_t consumed_size_;
  };
  union { Impl_ _impl_; };
  friend struct ::TableStruct_brpc_2fstreaming_5frpc_5fmeta_2eproto;
};
// ===================================================================


// ===================================================================

#ifdef __GNUC__
  #pragma GCC diagnostic push
  #pragma GCC diagnostic ignored "-Wstrict-aliasing"
#endif  // __GNUC__
// StreamSettings

// required int64 stream_id = 1;
inline bool StreamSettings::_internal_has_stream_id() const {
  bool value = (_impl_._has_bits_[0] & 0x00000001u) != 0;
  return value;
}
inline bool StreamSettings::has_stream_id() const {
  return _internal_has_stream_id();
}
inline void StreamSettings::clear_stream_id() {
  _impl_.stream_id_ = int64_t{0};
  _impl_._has_bits_[0] &= ~0x00000001u;
}
inline int64_t StreamSettings::_internal_stream_id() const {
  return _impl_.stream_id_;
}
inline int64_t StreamSettings::stream_id() const {
  // @@protoc_insertion_point(field_get:brpc.StreamSettings.stream_id)
  return _internal_stream_id();
}
inline void StreamSettings::_internal_set_stream_id(int64_t value) {
  _impl_._has_bits_[0] |= 0x00000001u;
  _impl_.stream_id_ = value;
}
inline void StreamSettings::set_stream_id(int64_t value) {
  _internal_set_stream_id(value);
  // @@protoc_insertion_point(field_set:brpc.StreamSettings.stream_id)
}

// optional bool need_feedback = 2 [default = false];
inline bool StreamSettings::_internal_has_need_feedback() const {
  bool value = (_impl_._has_bits_[0] & 0x00000002u) != 0;
  return value;
}
inline bool StreamSettings::has_need_feedback() const {
  return _internal_has_need_feedback();
}
inline void StreamSettings::clear_need_feedback() {
  _impl_.need_feedback_ = false;
  _impl_._has_bits_[0] &= ~0x00000002u;
}
inline bool StreamSettings::_internal_need_feedback() const {
  return _impl_.need_feedback_;
}
inline bool StreamSettings::need_feedback() const {
  // @@protoc_insertion_point(field_get:brpc.StreamSettings.need_feedback)
  return _internal_need_feedback();
}
inline void StreamSettings::_internal_set_need_feedback(bool value) {
  _impl_._has_bits_[0] |= 0x00000002u;
  _impl_.need_feedback_ = value;
}
inline void StreamSettings::set_need_feedback(bool value) {
  _internal_set_need_feedback(value);
  // @@protoc_insertion_point(field_set:brpc.StreamSettings.need_feedback)
}

// optional bool writable = 3 [default = false];
inline bool StreamSettings::_internal_has_writable() const {
  bool value = (_impl_._has_bits_[0] & 0x00000004u) != 0;
  return value;
}
inline bool StreamSettings::has_writable() const {
  return _internal_has_writable();
}
inline void StreamSettings::clear_writable() {
  _impl_.writable_ = false;
  _impl_._has_bits_[0] &= ~0x00000004u;
}
inline bool StreamSettings::_internal_writable() const {
  return _impl_.writable_;
}
inline bool StreamSettings::writable() const {
  // @@protoc_insertion_point(field_get:brpc.StreamSettings.writable)
  return _internal_writable();
}
inline void StreamSettings::_internal_set_writable(bool value) {
  _impl_._has_bits_[0] |= 0x00000004u;
  _impl_.writable_ = value;
}
inline void StreamSettings::set_writable(bool value) {
  _internal_set_writable(value);
  // @@protoc_insertion_point(field_set:brpc.StreamSettings.writable)
}

// -------------------------------------------------------------------

// StreamFrameMeta

// required int64 stream_id = 1;
inline bool StreamFrameMeta::_internal_has_stream_id() const {
  bool value = (_impl_._has_bits_[0] & 0x00000002u) != 0;
  return value;
}
inline bool StreamFrameMeta::has_stream_id() const {
  return _internal_has_stream_id();
}
inline void StreamFrameMeta::clear_stream_id() {
  _impl_.stream_id_ = int64_t{0};
  _impl_._has_bits_[0] &= ~0x00000002u;
}
inline int64_t StreamFrameMeta::_internal_stream_id() const {
  return _impl_.stream_id_;
}
inline int64_t StreamFrameMeta::stream_id() const {
  // @@protoc_insertion_point(field_get:brpc.StreamFrameMeta.stream_id)
  return _internal_stream_id();
}
inline void StreamFrameMeta::_internal_set_stream_id(int64_t value) {
  _impl_._has_bits_[0] |= 0x00000002u;
  _impl_.stream_id_ = value;
}
inline void StreamFrameMeta::set_stream_id(int64_t value) {
  _internal_set_stream_id(value);
  // @@protoc_insertion_point(field_set:brpc.StreamFrameMeta.stream_id)
}

// optional int64 source_stream_id = 2;
inline bool StreamFrameMeta::_internal_has_source_stream_id() const {
  bool value = (_impl_._has_bits_[0] & 0x00000004u) != 0;
  return value;
}
inline bool StreamFrameMeta::has_source_stream_id() const {
  return _internal_has_source_stream_id();
}
inline void StreamFrameMeta::clear_source_stream_id() {
  _impl_.source_stream_id_ = int64_t{0};
  _impl_._has_bits_[0] &= ~0x00000004u;
}
inline int64_t StreamFrameMeta::_internal_source_stream_id() const {
  return _impl_.source_stream_id_;
}
inline int64_t StreamFrameMeta::source_stream_id() const {
  // @@protoc_insertion_point(field_get:brpc.StreamFrameMeta.source_stream_id)
  return _internal_source_stream_id();
}
inline void StreamFrameMeta::_internal_set_source_stream_id(int64_t value) {
  _impl_._has_bits_[0] |= 0x00000004u;
  _impl_.source_stream_id_ = value;
}
inline void StreamFrameMeta::set_source_stream_id(int64_t value) {
  _internal_set_source_stream_id(value);
  // @@protoc_insertion_point(field_set:brpc.StreamFrameMeta.source_stream_id)
}

// optional .brpc.FrameType frame_type = 3;
inline bool StreamFrameMeta::_internal_has_frame_type() const {
  bool value = (_impl_._has_bits_[0] & 0x00000008u) != 0;
  return value;
}
inline bool StreamFrameMeta::has_frame_type() const {
  return _internal_has_frame_type();
}
inline void StreamFrameMeta::clear_frame_type() {
  _impl_.frame_type_ = 0;
  _impl_._has_bits_[0] &= ~0x00000008u;
}
inline ::brpc::FrameType StreamFrameMeta::_internal_frame_type() const {
  return static_cast< ::brpc::FrameType >(_impl_.frame_type_);
}
inline ::brpc::FrameType StreamFrameMeta::frame_type() const {
  // @@protoc_insertion_point(field_get:brpc.StreamFrameMeta.frame_type)
  return _internal_frame_type();
}
inline void StreamFrameMeta::_internal_set_frame_type(::brpc::FrameType value) {
  assert(::brpc::FrameType_IsValid(value));
  _impl_._has_bits_[0] |= 0x00000008u;
  _impl_.frame_type_ = value;
}
inline void StreamFrameMeta::set_frame_type(::brpc::FrameType value) {
  _internal_set_frame_type(value);
  // @@protoc_insertion_point(field_set:brpc.StreamFrameMeta.frame_type)
}

// optional bool has_continuation = 4;
inline bool StreamFrameMeta::_internal_has_has_continuation() const {
  bool value = (_impl_._has_bits_[0] & 0x00000010u) != 0;
  return value;
}
inline bool StreamFrameMeta::has_has_continuation() const {
  return _internal_has_has_continuation();
}
inline void StreamFrameMeta::clear_has_continuation() {
  _impl_.has_continuation_ = false;
  _impl_._has_bits_[0] &= ~0x00000010u;
}
inline bool StreamFrameMeta::_internal_has_continuation() const {
  return _impl_.has_continuation_;
}
inline bool StreamFrameMeta::has_continuation() const {
  // @@protoc_insertion_point(field_get:brpc.StreamFrameMeta.has_continuation)
  return _internal_has_continuation();
}
inline void StreamFrameMeta::_internal_set_has_continuation(bool value) {
  _impl_._has_bits_[0] |= 0x00000010u;
  _impl_.has_continuation_ = value;
}
inline void StreamFrameMeta::set_has_continuation(bool value) {
  _internal_set_has_continuation(value);
  // @@protoc_insertion_point(field_set:brpc.StreamFrameMeta.has_continuation)
}

// optional .brpc.Feedback feedback = 5;
inline bool StreamFrameMeta::_internal_has_feedback() const {
  bool value = (_impl_._has_bits_[0] & 0x00000001u) != 0;
  PROTOBUF_ASSUME(!value || _impl_.feedback_ != nullptr);
  return value;
}
inline bool StreamFrameMeta::has_feedback() const {
  return _internal_has_feedback();
}
inline void StreamFrameMeta::clear_feedback() {
  if (_impl_.feedback_ != nullptr) _impl_.feedback_->Clear();
  _impl_._has_bits_[0] &= ~0x00000001u;
}
inline const ::brpc::Feedback& StreamFrameMeta::_internal_feedback() const {
  const ::brpc::Feedback* p = _impl_.feedback_;
  return p != nullptr ? *p : reinterpret_cast<const ::brpc::Feedback&>(
      ::brpc::_Feedback_default_instance_);
}
inline const ::brpc::Feedback& StreamFrameMeta::feedback() const {
  // @@protoc_insertion_point(field_get:brpc.StreamFrameMeta.feedback)
  return _internal_feedback();
}
inline void StreamFrameMeta::unsafe_arena_set_allocated_feedback(
    ::brpc::Feedback* feedback) {
  if (GetArenaForAllocation() == nullptr) {
    delete reinterpret_cast<::PROTOBUF_NAMESPACE_ID::MessageLite*>(_impl_.feedback_);
  }
  _impl_.feedback_ = feedback;
  if (feedback) {
    _impl_._has_bits_[0] |= 0x00000001u;
  } else {
    _impl_._has_bits_[0] &= ~0x00000001u;
  }
  // @@protoc_insertion_point(field_unsafe_arena_set_allocated:brpc.StreamFrameMeta.feedback)
}
inline ::brpc::Feedback* StreamFrameMeta::release_feedback() {
  _impl_._has_bits_[0] &= ~0x00000001u;
  ::brpc::Feedback* temp = _impl_.feedback_;
  _impl_.feedback_ = nullptr;
#ifdef PROTOBUF_FORCE_COPY_IN_RELEASE
  auto* old =  reinterpret_cast<::PROTOBUF_NAMESPACE_ID::MessageLite*>(temp);
  temp = ::PROTOBUF_NAMESPACE_ID::internal::DuplicateIfNonNull(temp);
  if (GetArenaForAllocation() == nullptr) { delete old; }
#else  // PROTOBUF_FORCE_COPY_IN_RELEASE
  if (GetArenaForAllocation() != nullptr) {
    temp = ::PROTOBUF_NAMESPACE_ID::internal::DuplicateIfNonNull(temp);
  }
#endif  // !PROTOBUF_FORCE_COPY_IN_RELEASE
  return temp;
}
inline ::brpc::Feedback* StreamFrameMeta::unsafe_arena_release_feedback() {
  // @@protoc_insertion_point(field_release:brpc.StreamFrameMeta.feedback)
  _impl_._has_bits_[0] &= ~0x00000001u;
  ::brpc::Feedback* temp = _impl_.feedback_;
  _impl_.feedback_ = nullptr;
  return temp;
}
inline ::brpc::Feedback* StreamFrameMeta::_internal_mutable_feedback() {
  _impl_._has_bits_[0] |= 0x00000001u;
  if (_impl_.feedback_ == nullptr) {
    auto* p = CreateMaybeMessage<::brpc::Feedback>(GetArenaForAllocation());
    _impl_.feedback_ = p;
  }
  return _impl_.feedback_;
}
inline ::brpc::Feedback* StreamFrameMeta::mutable_feedback() {
  ::brpc::Feedback* _msg = _internal_mutable_feedback();
  // @@protoc_insertion_point(field_mutable:brpc.StreamFrameMeta.feedback)
  return _msg;
}
inline void StreamFrameMeta::set_allocated_feedback(::brpc::Feedback* feedback) {
  ::PROTOBUF_NAMESPACE_ID::Arena* message_arena = GetArenaForAllocation();
  if (message_arena == nullptr) {
    delete _impl_.feedback_;
  }
  if (feedback) {
    ::PROTOBUF_NAMESPACE_ID::Arena* submessage_arena =
        ::PROTOBUF_NAMESPACE_ID::Arena::InternalGetOwningArena(feedback);
    if (message_arena != submessage_arena) {
      feedback = ::PROTOBUF_NAMESPACE_ID::internal::GetOwnedMessage(
          message_arena, feedback, submessage_arena);
    }
    _impl_._has_bits_[0] |= 0x00000001u;
  } else {
    _impl_._has_bits_[0] &= ~0x00000001u;
  }
  _impl_.feedback_ = feedback;
  // @@protoc_insertion_point(field_set_allocated:brpc.StreamFrameMeta.feedback)
}

// -------------------------------------------------------------------

// Feedback

// optional int64 consumed_size = 1;
inline bool Feedback::_internal_has_consumed_size() const {
  bool value = (_impl_._has_bits_[0] & 0x00000001u) != 0;
  return value;
}
inline bool Feedback::has_consumed_size() const {
  return _internal_has_consumed_size();
}
inline void Feedback::clear_consumed_size() {
  _impl_.consumed_size_ = int64_t{0};
  _impl_._has_bits_[0] &= ~0x00000001u;
}
inline int64_t Feedback::_internal_consumed_size() const {
  return _impl_.consumed_size_;
}
inline int64_t Feedback::consumed_size() const {
  // @@protoc_insertion_point(field_get:brpc.Feedback.consumed_size)
  return _internal_consumed_size();
}
inline void Feedback::_internal_set_consumed_size(int64_t value) {
  _impl_._has_bits_[0] |= 0x00000001u;
  _impl_.consumed_size_ = value;
}
inline void Feedback::set_consumed_size(int64_t value) {
  _internal_set_consumed_size(value);
  // @@protoc_insertion_point(field_set:brpc.Feedback.consumed_size)
}

#ifdef __GNUC__
  #pragma GCC diagnostic pop
#endif  // __GNUC__
// -------------------------------------------------------------------

// -------------------------------------------------------------------


// @@protoc_insertion_point(namespace_scope)

}  // namespace brpc

PROTOBUF_NAMESPACE_OPEN

template <> struct is_proto_enum< ::brpc::FrameType> : ::std::true_type {};
template <>
inline const EnumDescriptor* GetEnumDescriptor< ::brpc::FrameType>() {
  return ::brpc::FrameType_descriptor();
}

PROTOBUF_NAMESPACE_CLOSE

// @@protoc_insertion_point(global_scope)

#include <google/protobuf/port_undef.inc>
#endif  // GOOGLE_PROTOBUF_INCLUDED_GOOGLE_PROTOBUF_INCLUDED_brpc_2fstreaming_5frpc_5fmeta_2eproto
