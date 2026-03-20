type SectionTitleProps = {
  children: React.ReactNode;
};

export function SectionTitle({ children }: SectionTitleProps) {
  return (
    <h2
      className="section-title-gradient mb-10 text-center font-sans text-[2.4rem] font-extrabold tracking-[0.01em] sm:text-[2.8rem]"
    >
      {children}
    </h2>
  );
}
