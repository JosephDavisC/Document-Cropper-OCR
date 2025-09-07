(function () {
  // render icons
  if (window.lucide && lucide.createIcons) lucide.createIcons();

  const $ = (id) => document.getElementById(id);
  const ok = (el) => el !== null && el !== undefined;

  const form = $('upload-form');
  const fileInput = $('file-input');
  const submitBtn = $('submit-btn');
  const dropzone = $('dropzone');
  const dzEmpty = $('dz-empty');
  const preview = $('preview');
  const previewImg = $('preview-img');
  const spinner = $('spinner');

  // Lightbox refs
  const lb = $('lightbox');
  const lbImg = $('lightbox-img');
  const lbClose = $('lb-close');

  // Ensure lightbox is hidden on initial load
  if (ok(lb)) lb.classList.add('hidden');

  function setReady(v) { if (ok(submitBtn)) submitBtn.disabled = !v; }

  function showPreview(file) {
    if (!ok(preview) || !ok(previewImg) || !ok(dzEmpty)) return;
    const url = URL.createObjectURL(file);
    previewImg.setAttribute('src', url);
    preview.classList.remove('hidden');
    dzEmpty.style.display = 'none';
    setReady(true);
  }

  // file input
  if (ok(fileInput)) {
    fileInput.addEventListener('change', (e) => {
      const f = e.target.files && e.target.files[0];
      if (f) showPreview(f);
    });
  }

  // drag & drop
  if (ok(dropzone)) {
    const stop = (e) => { e.preventDefault(); e.stopPropagation(); };
    ['dragenter','dragover','dragleave','drop'].forEach(ev => dropzone.addEventListener(ev, stop));
    ['dragenter','dragover'].forEach(ev => dropzone.addEventListener(ev, () => dropzone.classList.add('drag')));
    ['dragleave','drop'].forEach(ev => dropzone.addEventListener(ev, () => dropzone.classList.remove('drag')));
    dropzone.addEventListener('drop', (e) => {
      const dt = e.dataTransfer; if (!dt || !dt.files || !dt.files.length) return;
      const file = dt.files[0];
      if (ok(fileInput)) fileInput.files = dt.files;
      showPreview(file);
    });
  }

  // spinner on submit
  if (ok(form) && ok(spinner)) {
    form.addEventListener('submit', () => spinner.classList.remove('hidden'));
  }

  // Lightbox: open ONLY for elements explicitly marked data-lightbox="true"
  function openLB(src) {
    if (!src || !ok(lb) || !ok(lbImg)) return;
    lbImg.setAttribute('src', src);
    lb.classList.remove('hidden');
  }
  function closeLB() {
    if (ok(lb)) lb.classList.add('hidden');
    if (ok(lbImg)) lbImg.removeAttribute('src');
  }

  document.addEventListener('click', (e) => {
    const wrap = e.target.closest('.img-wrap[data-lightbox="true"]');
    if (!wrap) return;
    const img = wrap.querySelector('img');
    const src = img ? img.getAttribute('src') : null;
    if (src && src.length > 3) openLB(src);
  });

  if (ok(lbClose)) lbClose.addEventListener('click', closeLB);
  if (ok(lb)) lb.addEventListener('click', (e) => { if (e.target === lb) closeLB(); });
  document.addEventListener('keydown', (e) => { if (e.key === 'Escape') closeLB(); });
})();