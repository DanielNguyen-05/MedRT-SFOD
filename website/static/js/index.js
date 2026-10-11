// MedRT-SFSeg project page: carousel and BibTeX copy button.

document.addEventListener('DOMContentLoaded', function () {
  if (window.bulmaCarousel) {
    bulmaCarousel.attach('.carousel', {
      slidesToScroll: 1,
      slidesToShow: 3,
      loop: true,
      infinite: true,
      autoplay: true,
      autoplaySpeed: 3500,
      pauseOnHover: true,
      breakpoints: [
        { changePoint: 480, slidesToShow: 1, slidesToScroll: 1 },
        { changePoint: 768, slidesToShow: 2, slidesToScroll: 1 },
        { changePoint: 1024, slidesToShow: 3, slidesToScroll: 1 }
      ]
    });
  }

  var button = document.getElementById('copy-bibtex');
  var code = document.getElementById('bibtex-code');
  if (button && code) {
    button.addEventListener('click', function () {
      var label = button.querySelector('.copy-label');
      var done = function (text) {
        label.textContent = text;
        setTimeout(function () { label.textContent = 'Copy'; }, 1600);
      };
      var text = code.textContent;
      if (navigator.clipboard && navigator.clipboard.writeText) {
        navigator.clipboard.writeText(text).then(function () { done('Copied'); }, function () { done('Press Ctrl+C'); });
      } else {
        var range = document.createRange();
        range.selectNodeContents(code);
        var selection = window.getSelection();
        selection.removeAllRanges();
        selection.addRange(range);
        done('Press Ctrl+C');
      }
    });
  }
});
