# Feature Specification: Visitor Count

**Feature Branch**: `[008-visitor-count]`

**Created**: 2026-07-12

**Status**: Draft

**Input**: User description: "Add a visitor count endpoint that the Next.js
frontend calls once when the page loads. Keep it as a simple page-load counter
without client-side visitor or API checks."

## User Scenarios & Testing *(mandatory)*

### User Story 1 - Record And Display A Visit (Priority: P1)

As a website visitor, I want my page load to contribute to a visible total so I
can see how much traffic the demo has received.

**Why this priority**: Recording a page load and returning the updated total is
the complete minimum valuable behavior for this feature.

**Independent Test**: Begin with a known count, submit one valid visit, and
verify that the returned count and the subsequently observed count are exactly
one greater.

**Acceptance Scenarios**:

1. **Given** the stored count is 41, **When** the page records a visit, **Then**
   the visit succeeds and the updated count is 42.
2. **Given** the stored count is zero, **When** the first visit is recorded,
   **Then** the updated count is one.
3. **Given** the frontend loads the page once and makes one recording request,
   **When** that request succeeds, **Then** exactly one visit is added.

---

### User Story 2 - Preserve An Accurate Total (Priority: P2)

As a site operator, I need accepted visits to remain counted across restarts and
simultaneous traffic so the displayed total does not lose recorded page loads.

**Why this priority**: A total that decreases after restart or loses concurrent
updates would be misleading.

**Independent Test**: Record a known batch of visits concurrently, restart the
serving application, and verify the total increased by the batch size and
remains unchanged after restart.

**Acceptance Scenarios**:

1. **Given** multiple visits are submitted at the same time, **When** all are
   accepted, **Then** the total increases exactly once for every accepted
   request without lost updates.
2. **Given** visits have already been recorded, **When** the serving application
   restarts, **Then** the existing total is preserved.
3. **Given** a visit cannot be durably recorded, **When** it is submitted,
   **Then** the operation fails explicitly and does not report an increased
   total.

---

### User Story 3 - Support A Public Web Page Safely (Priority: P3)

As a frontend developer, I need the page to record a visit directly without
exposing private application credentials or supplying visitor identity data.

**Why this priority**: The browser must be able to use the feature safely, but
this is secondary to correct counting and persistence.

**Independent Test**: Submit a visit from an allowed public web origin without
credentials or visitor data and verify it succeeds, while malformed input and
disallowed browser origins are rejected.

**Acceptance Scenarios**:

1. **Given** the public page is loaded from an allowed origin, **When** it
   records a visit without private credentials, **Then** the visit succeeds.
2. **Given** a request includes an unsupported page identity or malformed
   shape, **When** it is submitted, **Then** it is rejected without changing
   the count.
3. **Given** a browser request originates from a disallowed origin, **When** it
   attempts to record a visit, **Then** browser access is not permitted.

### Edge Cases

- The persistent count store is missing, unreachable, read-only, or
  misconfigured when the application starts or a visit is submitted.
- Two or more recording requests read the same prior total concurrently.
- A request is retried by a browser or intermediary; each accepted request is a
  separate page view because deduplication is outside this feature.
- Development behavior causes the frontend to invoke its load effect more than
  once; every accepted request is counted under the same page-view semantics.
- The stored value is missing, malformed, negative, or exceeds the supported
  count range; the system fails explicitly rather than repairing or resetting
  it.
- A caller supplies visitor identifiers, alternate page representations, or
  additional unsupported fields; the request is rejected rather than
  normalized.
- Automated clients or bots call the public operation; bot detection and
  filtering are outside this feature.

## Requirements *(mandatory)*

### Functional Requirements

- **FR-001**: The system MUST provide a public operation that records one page
  view per accepted request without requiring private application credentials.
- **FR-002**: Recording a page view MUST increase the stored count by exactly
  one and return the resulting total.
- **FR-003**: The increment and retrieval of the resulting total MUST behave as
  one indivisible operation so simultaneous requests cannot lose updates or
  return a total older than their own increment.
- **FR-004**: The visitor count MUST persist across application restarts and
  deployments.
- **FR-005**: The system MUST support exactly one required page identity from a
  fixed supported set and MUST reject malformed, unsupported, alternate, or
  additional request fields.
- **FR-006**: A failed recording attempt MUST NOT report success or return a
  total that implies the failed visit was durably counted.
- **FR-007**: Missing or invalid count storage configuration, unavailable
  storage, and invalid stored state MUST produce explicit failures rather than
  an in-memory, reset, estimated, or silently repaired count.
- **FR-008**: The public operation MUST be usable by the configured frontend
  origin without exposing the existing private application token.
- **FR-009**: Each accepted request MUST count as one page view; the system MUST
  NOT infer unique people or deduplicate requests by session, cookie, network
  address, or other visitor identity.
- **FR-010**: The feature MUST NOT collect or persist visitor identifiers,
  network addresses, browser fingerprints, or other personal visitor data for
  counting purposes.
- **FR-011**: Unique-visitor analytics, bot filtering, rate limiting, historical
  time-series reporting, per-route dashboards, and frontend implementation MUST
  remain outside this feature.
- **FR-012**: The current authenticated serving operations MUST retain their
  existing authentication behavior; only the visitor-count operation is
  public.

### Architecture And Reuse Requirements *(mandatory)*

- **AR-001**: Visitor-count serving behavior MUST be owned by a dedicated
  feature surface in `packages/aitraf-api`; training, evaluation, and ML runtime
  packages MUST remain unchanged.
- **AR-002**: The feature MUST extend the existing application routing,
  configuration, and test surfaces rather than introduce a parallel serving
  application.
- **AR-003**: Count validation and increment-result transformation MUST use
  small, reusable functions with explicit inputs and outputs, while persistent
  state access remains localized at its boundary.
- **AR-004**: Production counting behavior MUST live in versioned repository
  code and MUST NOT depend on notebook-only or manual state changes.
- **AR-005**: The feature MUST not introduce process-local mutable state as the
  authoritative visitor count.
- **AR-006**: Tests, local configuration examples, API documentation, and
  validation commands MUST be updated directly for the new operation without
  compatibility aliases or alternate counting paths.
- **AR-007**: The feature MUST accept one required visit-recording schema and one
  required stored-count representation, rejecting alternate shapes and values
  rather than normalizing them.

### Validation And Reproducibility Requirements *(mandatory)*

- **VR-001**: Automated tests MUST cover the first visit, sequential visits,
  concurrent visits, persistence across restart, public access, disallowed
  origins, invalid request shapes, invalid stored state, and unavailable
  storage.
- **VR-002**: Command-level smoke validation MUST begin from a known total,
  record a visit through the deployed serving surface, verify an increase of
  exactly one, restart the application, and verify the increased total remains.
- **VR-003**: Verification MUST record the supported page identity, allowed
  frontend origin, persistent-store configuration, initial count, submitted
  request count, and commands required to reproduce the result.
- **VR-004**: A concurrent validation MUST prove that the final count increases
  by exactly the number of accepted requests with zero lost increments.
- **VR-005**: Failure validation MUST prove that malformed input, invalid stored
  state, and unavailable persistent storage fail explicitly without reporting a
  successful increment.

### Key Entities *(include if feature involves data)*

- **Page Visit**: One accepted page-load recording request for the single
  supported page identity. It contains no visitor identity or personal data.
- **Visitor Count**: The non-negative persistent lifetime total of accepted page
  visits and the resulting value returned after each successful increment.
- **Visitor Count Configuration**: Required runtime configuration identifying
  the persistent count store, the supported page identity, and the frontend
  origin permitted to use the public operation from a browser.

## Architecture And Data Impact

- **Touched Surfaces**: `packages/aitraf-api` feature routing, application
  composition, runtime configuration, tests, local dependency configuration,
  and API documentation. The separately deployed frontend and production
  deployment repository are external consumers and are not changed by this
  feature.
- **Shared Helpers To Add Or Extend**: Strict stored-count validation and a
  focused persistent increment boundary; these remain API-owned unless another
  package gains a genuine consumer.
- **Legacy Surfaces Removed**: None. The feature adds one new public capability
  and does not replace an existing visitor-count path.
- **Required Input Shapes**: A recording request contains exactly one supported
  page identity. The persisted value is one non-negative integer within the
  supported range. Missing, extra, alternate, malformed, negative, or
  out-of-range values fail explicitly.
- **Data Or Artifact Impact**: Adds one durable lifetime page-view total. Model,
  prediction, training, evaluation, media, and tracking artifacts are unchanged.
  No visitor-level records or personal data are retained.
- **Reproducibility Inputs**: Supported page identity, allowed frontend origin,
  persistent-store configuration, known starting total, application start and
  restart commands, and sequential and concurrent smoke requests.

## Success Criteria *(mandatory)*

### Measurable Outcomes

- **SC-001**: 100% of accepted recording requests increase the durable visitor
  count by exactly one and return the resulting total.
- **SC-002**: A validation run of at least 100 simultaneous accepted requests
  produces exactly 100 additional visits with zero lost increments.
- **SC-003**: After an application restart, the observed total equals the last
  successfully returned total in 100% of restart validation runs.
- **SC-004**: A page-load recording operation completes within one second for at
  least 95% of requests under normal demo traffic.
- **SC-005**: 100% of tested malformed requests and persistent-store failures
  are reported as failures and do not report an increased count.
- **SC-006**: The public page can record a visit without receiving or exposing
  any private application credential or visitor-identifying value.

## Assumptions

- “Visitor count” means a lifetime page-view count, not a unique-person count.
- The frontend calls the recording operation once after the target page loads;
  each accepted call is intentionally counted, including retries or duplicate
  development invocations.
- One named demo page is counted initially. Additional pages and per-page
  reporting require a future feature change.
- The frontend is deployed separately from this repository and will be updated
  independently to make the single page-load call and display the returned
  count.
- Production deployment configuration is owned by another repository and is an
  external follow-up, not an implementation surface of this feature.
- A durable external state service is available to all deployed API instances;
  its specific technology and operational design are selected during planning.
- Invalid or unavailable persistent state is surfaced as an explicit failure
  rather than replaced with an in-memory or default count.
