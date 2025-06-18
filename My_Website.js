const aboutMe = document.querySelector('.aboutme-page');
const initialTop = parseInt(window.getComputedStyle(aboutMe).top);

//listen for the scroll
window.addEventListener('scroll', () => {
    const scrollPosition = window.scrollY;
    const scrollStart = 50;
    if (scrollPosition > scrollStart) {
        aboutMe.style.transition = 'top 1.5s ease'
        aboutMe.style.top = '230px'
    } else {
        aboutMe.style.transition = 'top 1.5s ease'
        aboutMe.style.top = initialTop + 'px' }
});


function showSidebar() {
    const sidebar = document.querySelector('.side-bar')
    sidebar.style.display = 'flex'
}

function hideSidebar() {
    const sidebar = document.querySelector('.side-bar')
    sidebar.style.display = 'none'
}

//Multi-slider logic for three sections
function setupSlider(sliderSelector, prevSelector, nextSelector) {
    const slider = document.querySelector(sliderSelector);
    const slides = Array.from(slider.querySelectorAll('.slides'));
    const prev = slider.parentElement.querySelector(prevSelector);
    const next = slider.parentElement.querySelector(nextSelector);
    let current = 0;
    let isAnimating = false;

    function showSlide(idx, direction = 0) {
        if (isAnimating) return;
        isAnimating = true;
        slides.forEach((slide, i) => {
            slide.style.display = 'none';
            slide.classList.remove('slide-in-right', 'slide-out-left', 'slide-in-left', 'slide-out-right');
        });
        const currentSlide = slides[current];
        const nextSlide = slides[idx];
        if (direction === 1) { // next
            currentSlide.classList.add('slide-out-left');
            nextSlide.classList.add('slide-in-right');
        } else if (direction === -1) { // prev
            currentSlide.classList.add('slide-out-right');
            nextSlide.classList.add('slide-in-left');
        }
        nextSlide.style.display = 'flex';
        setTimeout(() => {
            slides.forEach((slide, i) => {
                slide.style.display = i === idx ? 'flex' : 'none';
                slide.classList.remove('slide-in-right', 'slide-out-left', 'slide-in-left', 'slide-out-right');
            });
            updateDots(idx);
            isAnimating = false;
        }, 400);
        current = idx;
    }

    // Create dot indicators
    let dotsContainer = slider.parentElement.querySelector('.slider-dots');
    if (!dotsContainer) {
        dotsContainer = document.createElement('div');
        dotsContainer.className = 'slider-dots';
        slides.forEach((_, idx) => {
            let dot = document.createElement('div');
            dot.className = 'slider-dot' + (idx === 0 ? ' active' : '');
            dot.addEventListener('click', () => {
                if (idx === current) return;
                showSlide(idx, idx > current ? 1 : -1);
            });
            dotsContainer.appendChild(dot);
        });
        slider.parentElement.appendChild(dotsContainer);
    }
    function updateDots(idx) {
        const dots = dotsContainer.querySelectorAll('.slider-dot');
        dots.forEach((dot, i) => {
            dot.classList.toggle('active', i === idx);
        });
    }

    next.addEventListener('click', function() {
        if (isAnimating) return;
        const nextIdx = (current + 1) % slides.length;
        showSlide(nextIdx, 1);
    });
    prev.addEventListener('click', function() {
        if (isAnimating) return;
        const prevIdx = (current - 1 + slides.length) % slides.length;
        showSlide(prevIdx, -1);
    });

    showSlide(current);
}

document.addEventListener('DOMContentLoaded', function () {
    setupSlider('.cs-slider', '.cs-buttons .prev', '.cs-buttons .next');
    setupSlider('.ee-slider', '.ee-buttons .prev', '.ee-buttons .next');
    setupSlider('.drawings-slider', '.drawings-buttons .prev', '.drawings-buttons .next');

    // Animate section titles on scroll (Apple-like)
    const sectionTitles = document.querySelectorAll('.section-title');
    const observer = new window.IntersectionObserver((entries) => {
        entries.forEach(entry => {
            if (entry.isIntersecting) {
                entry.target.classList.add('visible');
            }
        });
    }, { threshold: 0.6 });
    sectionTitles.forEach(title => observer.observe(title));

    const sidebarLinks = document.querySelectorAll('.side-bar a');
    const sidebar = document.querySelector('.side-bar');

    sidebarLinks.forEach(link => {
        link.addEventListener('click', function (event) {

            if (this.getAttribute('href').startsWith('mailto:')) {
                return;
            }
            event.preventDefault();
            
            hideSidebar();

            const targetId = this.getAttribute('href');

            document.querySelector(targetId).scrollIntoView({
                behavior: 'smooth'
            });
        });
    });
});

document.addEventListener('DOMContentLoaded', function () {
    const content = document.querySelectorAll('.slider .slides:nth-child(1) .content, .slider .slides:nth-child(2) .content, .slider .slides:nth-child(3) .content, .slider .slides:nth-child(4) .content');

    content.forEach(cont => {
        cont.addEventListener('touchstart', function () {
            this.classList.add('touch-active');
        });
        cont.addEventListener('touchend', function () {
            setTimeout(() => {
                this.classList.remove('touch-active');
            }, 200);
        });
    });
});


