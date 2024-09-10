## Example config file
```yaml
app:
    title: Realtime fMRI Motion
    session_secret:
        env: SCANBUDDY_SESSION_KEY
    auth:
        user: scanbuddy
        pass:
            env: SCANBUDDY_PASS
params:
    coil_elements:
        bad:
            - receive_coil: Head_32
              coil_elements:  HEA
            - receive_coil: Head_32
              coil_elements: HEP
        message: |
            Session: {SESSION}
            Series: {SERIES}
            Coil: {RECEIVE_COIL}, {COIL_ELEMENTS}
            
            Detected an issue with head coil elements.

            1. Check head coil connection for debris or other obstructions.
            2. Reconnect head coil securely.
            3. Ensure that anterior and posterior coil elements are present.

            Call 867-5309 for further assistance.
```
