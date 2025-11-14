# -*- mode: python ; coding: utf-8 -*-

block_cipher = None

a = Analysis(
    ['main.py'],
    pathex=[],
    binaries=[],
    datas=[
        # Đóng gói cascade & model vào bundle .app
        ('assets/haarcascade_frontalface_default.xml', 'assets'),
        ('models/resnet18_best_from_scratch.pth', 'models'),
    ],
    hiddenimports=[],
    hookspath=[],
    hooksconfig={},
    runtime_hooks=[],
    excludes=[],
    noarchive=False,
    optimize=0,
)

pyz = PYZ(a.pure, a.zipped_data, cipher=block_cipher)

exe = EXE(
    pyz,
    a.scripts,
    a.binaries,   # giữ nguyên
    a.datas,      # giữ nguyên
    [],
    name='main',
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=True,
    upx_exclude=[],
    runtime_tmpdir=None,
    console=False,              # app dạng windowed
    disable_windowed_traceback=False,
    argv_emulation=False,
    target_arch=None,
    codesign_identity=None,
    entitlements_file=None,
)

# Khai báo quyền camera & bundle id cho macOS (TCC)
info_plist = {
    'NSCameraUsageDescription': 'Ứng dụng cần dùng camera để nhận diện cảm xúc.',
    # Nếu sau này có thu âm thì mở dòng dưới:
    # 'NSMicrophoneUsageDescription': 'Ứng dụng cần micro để ghi âm.',
    'CFBundleIdentifier': 'com.khanh.emotion',  # đổi nếu muốn
}

app = BUNDLE(
    exe,
    name='main.app',
    icon=None,
    bundle_identifier='com.khanh.emotion',      # khớp với CFBundleIdentifier
    info_plist=info_plist,                      # QUAN TRỌNG: thêm plist
)
