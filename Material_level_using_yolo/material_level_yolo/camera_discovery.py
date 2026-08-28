"""Windows DirectShow camera discovery without optional Python dependencies."""

from __future__ import annotations

import json
import platform
import shutil
import subprocess


class CameraDiscoveryError(RuntimeError):
    """Raised when friendly-name camera enumeration is unavailable."""


_DIRECTSHOW_ENUMERATION_SCRIPT = r"""
[Console]::OutputEncoding = [System.Text.UTF8Encoding]::new($false)
$source = @'
using System;
using System.Collections.Generic;
using System.Runtime.InteropServices;
using System.Runtime.InteropServices.ComTypes;

[ComImport, Guid("29840822-5B84-11D0-BD3B-00A0C911CE86"),
 InterfaceType(ComInterfaceType.InterfaceIsIUnknown)]
interface ICreateDevEnum {
    [PreserveSig]
    int CreateClassEnumerator(
        [In] ref Guid type,
        out IEnumMoniker enumMoniker,
        int flags
    );
}

[ComImport, Guid("55272A00-42CB-11CE-8135-00AA004BB851"),
 InterfaceType(ComInterfaceType.InterfaceIsIUnknown)]
interface IPropertyBag {
    [PreserveSig]
    int Read(
        [MarshalAs(UnmanagedType.LPWStr)] string name,
        [MarshalAs(UnmanagedType.Struct)] ref object value,
        IntPtr errorLog
    );

    [PreserveSig]
    int Write(
        [MarshalAs(UnmanagedType.LPWStr)] string name,
        [MarshalAs(UnmanagedType.Struct)] ref object value
    );
}

public static class PolymerDirectShowCameraNames {
    public static string[] GetNames() {
        Guid systemDeviceEnum =
            new Guid("62BE5D10-60EB-11D0-BD3B-00A0C911CE86");
        Guid videoInputCategory =
            new Guid("860BB310-5D01-11D0-BD3B-00A0C911CE86");
        Type type = Type.GetTypeFromCLSID(systemDeviceEnum);
        object instance = Activator.CreateInstance(type);
        var names = new List<string>();
        try {
            IEnumMoniker enumerator;
            int result = ((ICreateDevEnum)instance).CreateClassEnumerator(
                ref videoInputCategory,
                out enumerator,
                0
            );
            if (result != 0 || enumerator == null) {
                return names.ToArray();
            }
            IMoniker[] monikers = new IMoniker[1];
            while (enumerator.Next(1, monikers, IntPtr.Zero) == 0) {
                object bagObject = null;
                Guid bagId = typeof(IPropertyBag).GUID;
                try {
                    monikers[0].BindToStorage(
                        null,
                        null,
                        ref bagId,
                        out bagObject
                    );
                    object value = "";
                    if (((IPropertyBag)bagObject).Read(
                            "FriendlyName",
                            ref value,
                            IntPtr.Zero
                        ) == 0) {
                        names.Add(Convert.ToString(value));
                    }
                } finally {
                    if (bagObject != null && Marshal.IsComObject(bagObject)) {
                        Marshal.ReleaseComObject(bagObject);
                    }
                    if (monikers[0] != null &&
                        Marshal.IsComObject(monikers[0])) {
                        Marshal.ReleaseComObject(monikers[0]);
                    }
                }
            }
            if (Marshal.IsComObject(enumerator)) {
                Marshal.ReleaseComObject(enumerator);
            }
        } finally {
            if (Marshal.IsComObject(instance)) {
                Marshal.ReleaseComObject(instance);
            }
        }
        return names.ToArray();
    }
}
'@
Add-Type -TypeDefinition $source
ConvertTo-Json -InputObject @(
    [PolymerDirectShowCameraNames]::GetNames()
) -Compress
"""


def list_directshow_camera_names() -> tuple[str, ...]:
    """Return DirectShow video-input friendly names in OpenCV index order."""
    if platform.system() != "Windows":
        raise CameraDiscoveryError(
            "Friendly-name camera discovery currently requires Windows DirectShow."
        )
    executable = shutil.which("powershell.exe") or shutil.which("pwsh")
    if executable is None:
        raise CameraDiscoveryError(
            "PowerShell is required for Windows DirectShow camera discovery."
        )
    try:
        completed = subprocess.run(
            [
                executable,
                "-NoLogo",
                "-NoProfile",
                "-NonInteractive",
                "-Command",
                _DIRECTSHOW_ENUMERATION_SCRIPT,
            ],
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
            timeout=20,
            check=False,
        )
    except (OSError, subprocess.SubprocessError) as exc:
        raise CameraDiscoveryError(
            f"Could not enumerate Windows DirectShow cameras: {exc}"
        ) from exc
    if completed.returncode != 0:
        details = completed.stderr.strip() or completed.stdout.strip()
        raise CameraDiscoveryError(
            "Windows DirectShow camera enumeration failed"
            + (f": {details}" if details else ".")
        )
    try:
        payload = json.loads(completed.stdout.lstrip("\ufeff").strip())
    except (json.JSONDecodeError, TypeError) as exc:
        raise CameraDiscoveryError(
            "Windows DirectShow camera enumeration returned invalid data."
        ) from exc
    if not isinstance(payload, list) or not all(
        isinstance(name, str) and name.strip() for name in payload
    ):
        raise CameraDiscoveryError(
            "Windows DirectShow camera enumeration returned invalid device names."
        )
    return tuple(name.strip() for name in payload)
