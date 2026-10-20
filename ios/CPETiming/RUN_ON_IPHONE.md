# Running the CPE timing app on the iPhone (Mac + free Apple ID)

This measures how fast our two models (the object detector and the arrival GRU) run on the iPhone.
No paid Apple Developer account and no Python setup needed. About 1.5 hours in total, most of it
the phone running on its own.

**You need:** the Mac, the iPhone and its USB cable, an Apple ID (a free one is fine).

---

## 1. Download the app and the models (5 min)

1. Open https://github.com/PES-112/CompositePerceptionEngine/actions/workflows/ios-timing.yml
   (sign in to GitHub; downloads need an account).
2. Click the newest run with a **green tick**.
3. Scroll to **Artifacts** and download both:
   - **CPETiming-unsigned-ipa**: the app
   - **coreml-exports**: the two models on their own (for step 4)
4. Unzip both (double-click).

## 2. Install the app on the iPhone (10 min)

1. Install **Sideloadly** for macOS from https://sideloadly.io and open it.
2. Plug the iPhone into the Mac. On the phone, tap **Trust** and enter the passcode.
3. Drag `CPETiming-unsigned.ipa` into Sideloadly. Enter your Apple ID and press **Start**.
   Enter the password if asked; Apple may send a code to your devices.
4. On the iPhone:
   - Settings → Privacy & Security → **Developer Mode** → On. The phone restarts; confirm **Turn On**.
   - Settings → General → **VPN & Device Management** → tap your Apple ID → **Trust**.
5. **CPE Timing** is now on the home screen. With a free Apple ID it stops opening after 7 days;
   to use it again, repeat step 2.3.

## 3. Run the timing (about 45 min, mostly hands-off)

Set up so the numbers are fair:
- Charge the phone to **above 80%**, then **unplug it**.
- Close all other apps. Take the case off. Keep the phone at room temperature, not in the sun or on a
  warm laptop.
- Turn off Low Power Mode.

Then:
1. Open **CPE Timing**.
2. Tap **Benchmark: warm-up + 10 blocks**. Wait until the Status section says it's finished (about 2 min).
3. Tap it again for a second run.
4. Check the slider says **Sustained run: 30 min**, then tap **Start sustained run**. Put the phone
   down and don't touch it until it finishes. The screen stays on by itself.
5. Get the CSV files to the Mac: open the **Files** app → On My iPhone → **CPE Timing**, select all
   the `.csv` files and AirDrop them. The app's **Share** button under Status only shares the latest
   run, so Files is quicker.

## 4. Apple's own timing for the models (10 min, needs Xcode)

This gives a second opinion on the speed, measured by Apple's tools instead of our app.

1. Install **Xcode** from the Mac App Store if it isn't there (it's big; start the download early).
2. Keep the iPhone plugged in, unlocked, with Developer Mode on.
3. In the unzipped **coreml-exports** folder, find `CPEDetector.mlpackage` and double-click it.
   Xcode opens it.
4. Click the **Performance** tab → **+** → choose the iPhone → **Run** (leave the other settings as they are).
5. Take a screenshot of the result: the median prediction time and the compute-unit breakdown
   (CPU / GPU / Neural Engine).
6. Do the same for `CPEArrivalGRU.mlpackage`.

## 5. Send back

- All the **CSV files** from step 3 (two benchmark runs and one sustained run).
- The **two screenshots** from step 4.
- The phone **model** and **iOS version** (Settings → General → About).

Zip them and send them over.

## If something goes wrong

- **Sideloadly says "Guru Meditation" or a login error:** open Sideloadly's advanced options, set
  "Apple ID sideload" mode, and try again. Or sign in at appleid.apple.com once in a browser.
- **"Untrusted Developer" when opening the app:** do the VPN & Device Management step in 2.4.
- **No Developer Mode option in Settings:** install once with Sideloadly first (step 2.3); the
  option then appears.
- **No green run on GitHub, or the artifacts are missing:** tell us; we'll fix the build.
- **The app crashes or shows an error:** send a screenshot.
