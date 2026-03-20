export function Marquee() {
  return (
    <div className="w-[140%] overflow-hidden px-[1280px] py-[90px]" aria-hidden="true">
      <div className="section-title-gradient text-[90px] font-[500] tracking-[0.34em] opacity-70 lg:text-[60px]">
        <div className="inline-block whitespace-nowrap animate-marquee">
          BAMP
        </div>
      </div>
    </div>
  );
}
