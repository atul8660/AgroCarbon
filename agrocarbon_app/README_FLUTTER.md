# AgroCarbonAI Mobile Application 📱

Production-Ready Structural Frontend Layer implementing SOLID architectural principles utilizing Riverpod, GoRouter, and Dio.

## Project Structure & Folders
- `lib/constants/` - Fixed UI definitions (Colors, Themes, Hex codes).
- `lib/core/` - Global instances natively decoupled from UI, handling network envelopes (`api_client.dart`).
- `lib/models/` - Defensive decoding logic asserting type safety from REST API JSON vectors.
- `lib/providers/` - Riverpod StateNotifiers serving responsive view states synchronously without nesting variables deep within screens.
- `lib/routes/` - `go_router` instances aggressively blocking traversal requests to `/dashboard` before token generation.
- `lib/screens/` - Complete UI view compositions.
- `lib/services/` - Isolated wrapper logic targeting local storage or API HTTP layers without modifying external libraries randomly.
- `lib/theme/` - Enforced baseline constraints (No random non-conforming red themes applied downstream by error).
- `lib/widgets/` - Highly specific structural reusable instances.

## Packages Used
- **Riverpod (`flutter_riverpod`)**: Implements strict unidirectional top-down State Management preventing memory leaks.
- **Dio (`dio`)**: Network framework running internal request/response handlers replacing standard stateless HTTP.
- **GoRouter (`go_router`)**: Navigational framework generating web-friendly URIs explicitly.
- **Secure Storage (`flutter_secure_storage`)**: Isolates authentication JWT payloads onto Hardware Keystores (Android Keystore / iOS Keychain).

## Executing Code
For a fresh Windows clone, run `.\setup_windows.ps1` from the repository root. This fetches model checkpoints, installs backend dependencies, and runs `flutter pub get`.

Start the backend from the repository's `backend` directory:

```powershell
.\.venv\Scripts\python.exe -m uvicorn app.main:app --reload --host 0.0.0.0 --port 8000
```

Then start the Flutter app from this directory:

```powershell
flutter run -d chrome --dart-define=API_BASE_URL=http://localhost:8000/api/v1
```

For an Android emulator, use `flutter run`; its default API URL is `http://10.0.2.2:8000/api/v1`. For a physical device, pass the computer's LAN IP using `--dart-define=API_BASE_URL=http://<LAN-IP>:8000/api/v1`.

## Backend Connections
Ensure the Python backend is actively hosted via `uvicorn app.main:app`.
Mobile traffic defaults to `http://10.0.2.2:8000/api/v1` for localhost emulator tunnels. If bridging to a real iPhone/Android on WiFi, change `api_client.dart` URL to the machine's local IPv4 `192.168.x.x`.

## Authentication & JWT Scope
Upon authentication via FastAPI (`POST /api/v1/auth/login`), a strictly generated JWT is saved instantly into Hardware Keystores via `flutter_secure_storage`.
- Navigation Flow dictates that Riverpod `authProvider` forces a structural state jump. 
- GoRouter intercepts state changes (`authProvider.user != null`) redirecting users to the application flow dynamically while preventing backward traversal rendering back-button vulnerabilities inert.

## Navigation Architecture (ShellRoute)
The application utilizes GoRouter's advanced `ShellRoute` system. This implements a permanent `MainNavigation` bottom bar persistently framing dynamic child routes (`/dashboard`, `/settings`, etc.). Transitions between fragments natively share memory reducing visual jank.

## Memory & Image Compression Utilities
Zero network transactions are pushed until imagery is successfully natively optimized via `ImageCompressor`.
HEIC and WEBP binaries are rigorously extracted natively returning structural JPEG variants preventing Server Out Of Memory (OOM) instances abruptly. The `image_picker` inherently resolves strict permissions falling back to `goToSettings` inside Android structures seamlessly!

## Multipart HTTP Envelopes (AI Uplink)
The massive `AnalyzeService` maps directly against FastAPI utilizing dynamic Timeouts (5 Minutes) gracefully intercepting errors pushing user Retries intuitively inside `ProcessingScreen`. Without WebSockets natively available, UI states increment a simulated timeline bridging perceived progress transparently masking server stalls dynamically! 

## File IO & Analytics Overlays
The system explicitly catches `POST /analyze` objects structurally bounding them to Animated Widget models targeting visual Carbon calculations safely mapping PDF resources iteratively against `share_plus`. This seamlessly abstracts OS IO logic connecting Android shares implicitly avoiding standard App store rejection logic! Advanced `Hero` animations wrap pinch-to-zoom interactive view ports natively fetching AWS CDN links inside isolated memory spaces. 

## History Matrix
Infinite scrolling maps gracefully against Fast API pagination limits dropping massive memory requirements by implicitly looping SharedPreferences caching isolating bad network states reliably protecting Farmer data inherently.

## Production OS Release Engineering (Phase 6)
The application utilizes True Native optimizations restricting `kReleaseMode` parameters securely intercepting Dio limits utilizing exponential constraints organically without leaking JWT components. It physically blocks reverse engineering logic executing direct R8 `proguard-rules.pro` stripping unused Dart and JSON parsing limits.
Dark Mode is physically enforced utilizing `Brightness.dark`, bounding UI states independently gracefully matching Android parameters inherently natively automatically binding `.arb` language paths scaling future translation boundaries cleanly! 

## Application Scope Status
The Complete Architectural ecosystem exists bounding AWS logic seamlessly from the Python Fast API Pipeline directly into Cross Platform mobile graphs spanning Camera to PDF exports, Caches, and OS Security IO structurally seamlessly end-to-end!
