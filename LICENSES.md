# Mixed-license scope — not wholly open source

This repository currently has mixed licensing:

- The root [`LICENSE`](LICENSE) is the MIT License for repository content not
  covered by a more specific license.
- [`3d_camera/LICENSE`](3d_camera/LICENSE) applies to the entire `3d_camera/`
  directory and currently grants no permission to copy, modify, distribute,
  sublicense, or use that component without prior written permission from the
  copyright holder. It takes precedence over the root license for that path.
- Model weights, datasets, recordings, pretrained checkpoints, and other
  separately distributed research assets retain their own terms. They are not
  licensed merely because a configuration references them.
- Dependencies such as Intel RealSense, PyTorch, Ultralytics, OpenCV, and their
  transitive packages retain their respective upstream licenses and terms.

The integrated install does not erase these boundaries. Before a public
release, institutional or legal review should determine whether the D405
component can be relicensed and whether every model/data/dependency use is
compatible with the intended research and distribution model. Record the
decision in release notes; do not silently replace a restrictive license.

Until that review authorizes a change, do not describe the complete repository
or integrated distribution as MIT-licensed or open source. Researchers may use
the permissively licensed portions under the root MIT terms, but they must
obtain prior written permission for the `3d_camera/` component.

This file documents repository scope and is not legal advice.
